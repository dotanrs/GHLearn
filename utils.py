class conv_layer:

  def __init__(self, filter_size, num_filters, pool_size = 0):
    self.filter_size = filter_size
    self.num_filters = num_filters
    self.pool_size = pool_size

  def __str__(self):
    return ("[%d, %d, %d]" % (self.filter_size, self.num_filters, self.pool_size))