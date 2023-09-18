def google_config():
    return {
        'GOOGLE_CLIENT_ID': 'xxxxxxxxxxxxxxxxxxxxxxx',
        'GOOGLE_CLIENT_SECRET': 'xxxxxxxxxxxxxxxxxxxx',
        'GOOGLE_ENV_LOCAL': True,
        'GOOGLE_SCOPE': ['profile', 'email'],
    }


def mail_config():
    return {
        'MAIL_SERVER': 'smtp.gmail.com',
        'MAIL_PORT': 587,
        'MAIL_USERNAME': 'xxxxxxxxxx',
        'MAIL_PASSWORD': 'xxxxxxxxxx',
        'MAIL_USE_TLS': True,
        'MAIL_USE_SSL': False,
    }


def database_config():
    return {
        'USERNAME': 'xxxx',
        'PASSWORD': 'xxxxxx',
        'HOST': 'localhost',
        'DATABASE': 'xxxxxx'
    }


