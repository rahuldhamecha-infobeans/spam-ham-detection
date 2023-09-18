def google_config():
    return {
        'GOOGLE_CLIENT_ID': '110114966756-ft8iucu8mf6lj6jlnq5pmjko3qph59i9.apps.googleusercontent.com',
        'GOOGLE_CLIENT_SECRET': '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',
        'GOOGLE_ENV_LOCAL': True,
        'GOOGLE_SCOPE': ['profile', 'email'],
    }


def mail_config():
    return {
        'MAIL_SERVER': 'smtp.gmail.com',
        'MAIL_PORT': 587,
        'MAIL_USERNAME': 'rahul.dhamecha@infobeans.com',
        'MAIL_PASSWORD': 'cvmrwahrucmdolws',
        'MAIL_USE_TLS': True,
        'MAIL_USE_SSL': False,
    }


def database_config():
    return {
        'USERNAME': 'phpmyadmin',
        'PASSWORD': 'Root%401234',
        'HOST': 'localhost',
        'DATABASE': 'flask_17sept'
    }
