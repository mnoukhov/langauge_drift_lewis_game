from drift.core import eval_comm_loop, eval_listener_loop, eval_speaker_loop
import wandb


def _get_stats(names, game, speaker, listener):
    stats = {}
    s_stats, s_conf_mat = eval_speaker_loop(game.get_generator(1000, names), speaker)
    stats.update(s_stats)
    l_stats, l_conf_mat = eval_listener_loop(game.get_generator(1000, names), listener)
    stats.update(l_stats)
    comm_stats, l_conf_mat_gr_msg = eval_comm_loop(
        game.get_generator(1000, names), listener, speaker
    )
    stats.update(comm_stats)
    return stats, {
        "speak": s_conf_mat,
        "listen": l_conf_mat,
        "listen_gr_msg": l_conf_mat_gr_msg,
    }


def eval_loop(listener, speaker, game, writer, step, vocab_change_data, prefix=None):
    """Evaluation loop. Separate sp and heldout"""
    speaker.train(False)
    listener.train(False)

    # Record stats for SP
    sp_stats, sp_conf_mat = _get_stats("sp", game, speaker, listener)

    # Record stats, for heldout
    ho_stats, ho_conf_mat = _get_stats("heldout", game, speaker, listener)

    logs = {}

    # Plot conf mats
    for conf_mat_name in sp_conf_mat.keys():
        final_conf_mat = sp_conf_mat[conf_mat_name] * game.sp_ratio + ho_conf_mat[
            conf_mat_name
        ] * (1 - game.sp_ratio)
        if writer is not None:
            writer.add_image(conf_mat_name, final_conf_mat.unsqueeze(0), step)
        else:
            logs[conf_mat_name] = wandb.Image(final_conf_mat.unsqueeze(0))
        if vocab_change_data.get(conf_mat_name) is None:
            vocab_change_data[conf_mat_name] = []
        vocab_change_data[conf_mat_name].append(final_conf_mat)

    logstr = [f"[{prefix or ''}SP] step {step}:"]
    for name, val in sp_stats.items():
        logstr.append("{}: {:.4f}".format(name, val))
        if writer is not None:
            writer.add_scalar("sp/" + name, val, step)
        else:
            logs[f"sp/{name}"] = val
    # writer.flush()
    print(" ".join(logstr))

    logstr = [f"[{prefix or ''}HO] step {step}:"]
    for name, val in ho_stats.items():
        logstr.append("{}: {:.4f}".format(name, val))
        if writer is not None:
            writer.add_scalar("ho/" + name, val, step)
        else:
            logs[f"ho/{name}"] = val
    # writer.flush()
    print(" ".join(logstr))

    if prefix is not None:
        logs = {f"{prefix}/{key}": value for key, value in logs.items()}

    wandb.log(logs, step=step)
