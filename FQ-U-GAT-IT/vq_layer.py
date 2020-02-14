import tensorflow as tf
from tensorflow.python.training import moving_averages

class VectorQuantizerEMA:
  """Sonnet module representing the VQ-VAE layer.

  Args:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms (see
      equation 4 in the paper).
    decay: float, decay for the moving averages.
    epsilon: small float constant to avoid numerical instability.
  """

  def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay,
               epsilon=1e-5, name='VectorQuantizerEMA'):
    # super(VectorQuantizerEMA, self).__init__(name=name)
    self._embedding_dim = embedding_dim
    self._num_embeddings = num_embeddings
    self._decay = decay
    self._commitment_cost = commitment_cost
    self._epsilon = epsilon


  def __call__(self, inputs, reuse=False, layer=None, is_training=True):
    """Connects the module to some inputs.

    Args:
      inputs: Tensor, final dimension must be equal to embedding_dim. All other
        leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data. When
        this is set to False, the internal moving average statistics will not be
        updated.

    Returns:
      dict containing the following keys and values:
        quantize: Tensor containing the quantized version of the input.
        loss: Tensor containing the loss to optimize.
        perplexity: Tensor containing the perplexity of the encodings.
        encodings: Tensor containing the discrete encodings, ie which element
          of the quantized space each input element was mapped to.
        encoding_indices: Tensor containing the discrete encoding indices, ie
          which element of the quantized space each input element was mapped to.
    """
    # Ensure that the weights are read fresh for each timestep, which otherwise
    # would not be guaranteed in an RNN setup. Note that this relies on inputs
    # having a data dependency with the output of the previous timestep - if
    # this is not the case, there is no way to serialize the order of weight
    # updates within the module, so explicit external dependencies must be used.
    with tf.variable_scope('vq_layer%d'%layer, reuse=reuse):
      initializer = tf.random_normal_initializer()
      # w is a matrix with an embedding in each column. When training, the
      # embedding is assigned to be the average of all inputs assigned to that
      # embedding.
      self._w = tf.get_variable(
          'embedding', [self._embedding_dim, self._num_embeddings],
          initializer=initializer, use_resource=True)
      self._ema_cluster_size = tf.get_variable(
          'ema_cluster_size', [self._num_embeddings],
          initializer=tf.constant_initializer(0), use_resource=True)
      self._ema_w = tf.get_variable(
          'ema_dw', initializer=self._w.initialized_value(), use_resource=True)

      with tf.control_dependencies([inputs]):
        w = self._w.read_value()
      input_shape = tf.shape(inputs)
      with tf.control_dependencies([
          tf.Assert(tf.equal(input_shape[-1], self._embedding_dim),
                    [input_shape])]):
        flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])

      distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                   - 2 * tf.matmul(flat_inputs, w)
                   + tf.reduce_sum(w ** 2, 0, keepdims=True))

      encoding_indices = tf.argmax(- distances, 1)
      encodings = tf.one_hot(encoding_indices, self._num_embeddings)
      encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
      quantized = self.quantize(encoding_indices)
      e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)

      if is_training:
        updated_ema_cluster_size = moving_averages.assign_moving_average(
            self._ema_cluster_size, tf.reduce_sum(encodings, 0), self._decay)
        dw = tf.matmul(flat_inputs, encodings, transpose_a=True)
        updated_ema_w = moving_averages.assign_moving_average(self._ema_w, dw,
                                                              self._decay)
        n = tf.reduce_sum(updated_ema_cluster_size)
        updated_ema_cluster_size = (
            (updated_ema_cluster_size + self._epsilon)
            / (n + self._num_embeddings * self._epsilon) * n)

        normalised_updated_ema_w = (
            updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1]))
        with tf.control_dependencies([e_latent_loss]):
          update_w = tf.assign(self._w, normalised_updated_ema_w)
          with tf.control_dependencies([update_w]):
            loss = self._commitment_cost * e_latent_loss

      else:
        loss = self._commitment_cost * e_latent_loss
      quantized = inputs + tf.stop_gradient(quantized - inputs)
      avg_probs = tf.reduce_mean(encodings, 0)
      perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))

      return loss, perplexity

  @property
  def embeddings(self):
    return self._w

  def quantize(self, encoding_indices):
    with tf.control_dependencies([encoding_indices]):
      w = tf.transpose(self.embeddings.read_value(), [1, 0])
    return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)
