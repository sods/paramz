#
# This loads the configuration
#
import os
try:
    #Attempt Python 2 ConfigParser setup
    import ConfigParser
    config = ConfigParser.ConfigParser()
    from ConfigParser import NoOptionError
except ImportError:
    #Attempt Python 3 ConfigParser setup
    import configparser
    config = configparser.ConfigParser()
    from configparser import NoOptionError
    

# This is the default configuration file that always needs to be present.
default_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'defaults.cfg'))

# This file is optional and specifies configurations specific to the user (it is found in the user home directory i.e, ~/.config/paramz)
home = os.getenv('HOME') or os.getenv('USERPROFILE')
user_file = os.path.join(home,'.config','paramz', 'user.cfg')

# Read in the given files.
config.readfp(open(default_file))
config.read([user_file])

if not config:
    raise ValueError("No configuration file found at either " + user_file + " or " + default_file + ".")
