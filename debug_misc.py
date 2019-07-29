def debug_policy_plot():
    qq_right = []
    x_vec = np.arange(vagent.max_q[0])
    for xx in x_vec:
            vagent.q[0] = xx
            vsensor.update(vscene,vagent)
            vobservation  = local_observer(vsensor,vagent) #todo: generalize
            qq_right.append(RL.compute_q_eval(vobservation.reshape([1,-1])))
    qq_right = (np.reshape(qq_right,[-1,2]))
    vax[0].clear()
    vax[0].plot(qq_right, 'x-')
    vax[1].clear()
    vax[1].plot(qq_right[:,1]-qq_right[:,0], 'x-')