def get_name(opts, func='FUS'):
    if func == 'FUS':
        name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
            opts.score,
            opts.data_name,
            opts.model_name,
            opts.attack_name,
            opts.trigger,
            opts.target,
            opts.ratio,
            opts.n_iter
        )
        if opts.n_iter != 0:
            name = name + '_{}'.format(opts.alpha)
    elif func == 'BS':
        name = 'BS_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            opts.score,
            opts.data_name,
            opts.model_name,
            opts.attack_name,
            opts.trigger,
            opts.target,
            opts.ratio,
            opts.early_epoch
        )
    return name
