import numpy as np

def add_common_args(parser):
    parser.add_argument('--seed', '-s', type=int, default=np.random.randint(0, 1e6))
    parser.add_argument('--env_type', type=str, help='Set Perlin, Step or Degenrate environment type. Default: step', choices={'perlin', 'step', 'degenerate', 'stationary_reach', 'stationary_stay'}, default='step')
    parser.add_argument('--test_env_type', type=str, help='Set Perlin, Step or Degenrate environment type. Default: step', choices={'perlin', 'step', 'degenerate', 'stationary_reach', 'stationary_stay'}, default='step')
    parser.add_argument('--env_mode', type=str, help='Set enviornment state control mode - direct or with dynamics. Default: state (direct)', choices={'state', 'velocity', 'acceleration'}, default='state')
    parser.add_argument('--act_mode', type=str, help='Set relative or absolute action mode. Defauilt: relative', choices={'absolute', 'relative'}, default='relative')
    parser.add_argument('--perlin_discontinuous', help='If env_type is perlin, set mode to discontinuous with this flag', action='store_true')
    parser.add_argument('--decay_ac', help='', action='store_true')

    parser.add_argument('--reg_a', help='', action='store_true')
    parser.add_argument('--reg_s', help='', action='store_true')
    parser.add_argument('--s_eps', type=float, help='', default=0.05)
    parser.add_argument('--save', help='', action='store_true')

    parser.add_argument('--deterministic', '-d', help='Use determinisitic agents? - only affects PPO, TRPO and SAC', action='store_true')
    parser.add_argument('--test_target', '-T', help='(For off policy algs only) Test target policy?', action='store_true')

    parser.add_argument('--test_period', '-tp', type=int, help='How often to test agent and display output', default=20)
    parser.add_argument('--action_distr_test_period', '-adtp', type=int, help='How often to test the action distribution', default=100)
    parser.add_argument('--compare_filtered', help='Compare filtered vs non-filtered actions?', action='store_true')

    parser.add_argument('--hid', type=int, default=400)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=100)

    return parser

