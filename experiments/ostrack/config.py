
# #========================== Memnet ===============================
# hidden_size = 384  # pre: 512
# memory_size_pos = 8
# memory_size_neg = 16
# neg_dist_thre = 4
# neg_score_ratio = 0.7
# neg_num_write = 2

# slot_size = [8, 8, 384]
# # slot_size = [6, 6, 256]
# usage_decay = 0.99

# clip_gradients = 20.0
# keep_prob = 0.8
# weight_decay = 0.0001
# use_attention_read = False
# use_fc_key = False
# key_dim = 384  #  pre: 256


class config:
    hidden_size = 384  # pre: 512
    memory_size_pos = 8
    memory_size_neg = 16
    neg_dist_thre = 4
    neg_score_ratio = 0.7
    neg_num_write = 2

    slot_size = [8, 8, 384]
    # slot_size = [6, 6, 256]
    usage_decay = 0.99

    clip_gradients = 20.0
    keep_prob = 0.8
    weight_decay = 0.0001
    use_attention_read = False
    use_fc_key = False
    key_dim = 384  # pre: 256
    summary_display_step = 8


