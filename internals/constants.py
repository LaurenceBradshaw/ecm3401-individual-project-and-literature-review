

# Numerical constants
EARTH_ROTATION_SPEED = 7.292115e-5  # rad/s
SPEED_OF_LIGHT = 299792458.0        # m/s
C = SPEED_OF_LIGHT
L1_MIN = 1.55e9
L1_MAX = 1.61e9

# Constants for column names
PR_COL = 'CorrectedPseudorange'
PRR_COL = 'CorrectedPseudorangeRateMetersPerSecond'
SAT_POS_COLS = ['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']
SAT_VEL_COLS = ['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']

# Constellation mapping
CONSTELLATION_MAP = {
    0: 'UNK',
    1: 'G',
    2: 'SBAS',
    3: 'GLO',
    4: 'QZ',
    5: 'BD',
    6: 'GAL',
    7: 'I'
}