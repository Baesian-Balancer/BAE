import rl_smoothness.envs.StepEnv as StepEnv
import rl_smoothness.envs.PerlinEnv as PerlinEnv
import rl_smoothness.envs.StationaryEnv as StationaryEnv
from rl_smoothness.envs.DegEnv import DegEnv

def GetEnv(env_type='step', env_mode='state', args=dict(), **kwargs):
    if args['decay_ac']:
        decaying_ac = True
    else:
        decaying_ac = False

    if env_type == 'step':
        if env_mode == 'state':
            return StepEnv.StateEnv(decaying_ac=decaying_ac, **kwargs)
        elif env_mode == 'velocity':
            return StepEnv.VelocityEnv(decaying_ac=decaying_ac, **kwargs)
        elif env_mode == 'acceleration':
            return StepEnv.AccelerationEnv(decaying_ac=decaying_ac, **kwargs)
        else:
            raise NotImplementedError(env_mode + ' mode Step environment not yet implemented')
    elif env_type == 'perlin':
        if env_mode == 'state':
            if 'perlin_discontinuous' in args.keys():
                if args['perlin_discontinuous']:
                    continuous = False
                else:
                    continuous = True
            return PerlinEnv.StateEnv(continuous=continuous, decaying_ac=decaying_ac, **kwargs)
        elif env_mode == 'velocity':
            if 'perlin_discontinuous' in args.keys():
                if args['perlin_discontinuous']:
                    continuous = False
                else:
                    continuous = True
            return PerlinEnv.VelocityEnv(continuous=continuous, decaying_ac=decaying_ac, **kwargs)
        elif env_mode == 'acceleration':
            if 'perlin_discontinuous' in args.keys():
                if args['perlin_discontinuous']:
                    continuous = False
                else:
                    continuous = True
            return PerlinEnv.AccelerationEnv(continuous=continuous, decaying_ac=decaying_ac, **kwargs)
        else:
            raise NotImplementedError(env_mode + ' mode Perlin environment not yet implemented')
    elif env_type in ['stationary_reach', 'stationary_stay']:
        if env_type == 'stationary_reach': reach_reset=True
        else: reach_reset = False

        if env_mode == 'state':
            return StationaryEnv.StateEnv(reach_reset=reach_reset, **kwargs)
        elif env_mode == 'velocity':
            return StationaryEnv.VelocityEnv(reach_reset=reach_reset, **kwargs)
        elif env_mode == 'acceleration':
            return StationaryEnv.AccelerationEnv(reach_reset=reach_reset, **kwargs)
        else:
            raise NotImplementedError(env_mode + ' mode Step environment not yet implemented')

    elif env_type == 'degenerate':
        return DegEnv()
    else:
        raise NotImplementedError('env_type not recognized')