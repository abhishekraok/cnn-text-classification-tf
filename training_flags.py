class TrainingFlags:
    def __init__(self, embedding_dim=128, filter_sizes='3,4,5', num_epochs=1, l2_reg_lambda=0,
                 allow_soft_placement=True, log_device_placement=False, output_dir='.', num_checkpoints=3,
                 enable_word_embeddings=False, pretrained_embedding='.', evaluate_every=100, batch_size=64,
                 checkpoint_every=100, num_filters=128, decay_coefficient=2.5, dropout_keep_prob=0.5, is_word2vec=0,
                 min_learning_rate=0.0001, beta1=0.9, beta2=0.999, fully_connected_units=64):
        self.dropout_keep_prob = dropout_keep_prob
        self.decay_coefficient = decay_coefficient
        self.num_filters = num_filters
        self.evaluate_every = evaluate_every
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self.pretrained_embedding = pretrained_embedding
        self.num_checkpoints = num_checkpoints
        self.output_dir = output_dir
        self.enable_word_embeddings = enable_word_embeddings
        self.log_device_placement = log_device_placement
        self.allow_soft_placement = allow_soft_placement
        self.l2_reg_lambda = l2_reg_lambda
        self.filter_sizes = filter_sizes
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.is_word2vec = is_word2vec
        self.min_learning_rate = min_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.fully_connected_units = 64