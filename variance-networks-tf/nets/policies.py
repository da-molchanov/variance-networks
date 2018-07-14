def linear_decay(epoch_num, decay_start, total_epochs, start_value):
    if epoch_num < decay_start:
        return start_value
    return max(start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start), 1e-5)

lr_policy = lambda epoch_num: policies.linear_decay(
            epoch_num, decay_start=0, total_epochs=n_epochs, start_value=1e-3)

def step_decay(epoch_num, decay, decay_step, start_value):
    return start_value - (epoch_num/decay_step)*decay

def linear_growth(epoch_num, growth_start, total_epochs, start_value, end_value):
    if epoch_num < growth_start:
        return start_value
    return float(epoch_num-growth_start)/float(total_epochs-growth_start)*float(end_value-start_value)+start_value

