from torch.distributions import Categorical

BATCH_SIZE = 500


def selfplay_batch(
    objs, gumbel_temperature, l_opt, listener, s_opt, speaker, ent_coef=0.0
):
    """Generate a batch and play"""
    # Generate batch
    oh_msgs, entropy = speaker.new_gumbel(objs, gumbel_temperature)

    # Get gradient to keep backprop to speaker
    # oh_msgs = listener.one_hot(msgs)
    # oh_msgs.requires_grad = True
    # oh_msgs.grad = None
    l_logits = listener.get_logits(oh_msgs)

    # Train listener
    l_logprobs = Categorical(logits=l_logits).log_prob(objs)
    l_logprobs = l_logprobs.sum(-1)
    loss = -l_logprobs.mean()
    l_opt.zero_grad()
    s_opt.zero_grad()
    loss.backward()
    l_opt.step()
    s_opt.step()
