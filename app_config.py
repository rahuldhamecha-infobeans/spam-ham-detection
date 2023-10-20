import configparser

config = configparser.ConfigParser()
config.read('demo-config.ini')

def google_config():
    return {
        'GOOGLE_CLIENT_ID': config.get('Google', 'GOOGLE_CLIENT_ID'),
        'GOOGLE_CLIENT_SECRET': config.get('Google', 'GOOGLE_CLIENT_SECRET'),
        'GOOGLE_ENV_LOCAL': config.getboolean('Google', 'GOOGLE_ENV_LOCAL'),
        'GOOGLE_SCOPE': config.get('Google', 'GOOGLE_SCOPE').split(),
    }

def mail_config():
    return {
        'MAIL_SERVER': config.get('Mail', 'MAIL_SERVER'),
        'MAIL_PORT': config.getint('Mail', 'MAIL_PORT'),
        'MAIL_USERNAME': config.get('Mail', 'MAIL_USERNAME'),
        'MAIL_PASSWORD': config.get('Mail', 'MAIL_PASSWORD'),
        'MAIL_USE_TLS': config.getboolean('Mail', 'MAIL_USE_TLS'),
        'MAIL_USE_SSL': config.getboolean('Mail', 'MAIL_USE_SSL'),
    }

def database_config():
    return {
        'USERNAME': config.get('Database', 'USERNAME'),
        'PASSWORD': config.get('Database', 'PASSWORD'),
        'HOST': config.get('Database', 'HOST'),
        'DATABASE': config.get('Database', 'DATABASE'),
    }