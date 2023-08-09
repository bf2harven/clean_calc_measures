import os
import requests
from logging import Handler, Formatter
import logging
import datetime
from dateutil import tz


def send_email(message: str, email_address: str = 'shalom.rochman@mail.huji.ac.il',
               subject: str = "auto_mail_python", error: bool = False):
    os.system(f'echo {message} | mail {email_address} -s "{subject}{": error" if error else ": no error"}"')


loggers = dict()
def notify_telegram_bot(message, telegram_chat_id='59039965',
                        telegram_token='1782044343:AAExorKyiZRHtgLcSuSQQRgajt0mkKx0d3w', error: bool = False):
    class RequestsHandler(Handler):

        def __init__(self, telegram_chat_id, telegram_token):
            super().__init__()
            self.telegram_chat_id = telegram_chat_id
            self.telegram_token = telegram_token

        def emit(self, record):
            log_entry = self.format(record)
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': log_entry,
                'parse_mode': 'HTML'
            }
            return requests.post("https://api.telegram.org/bot{token}/sendMessage".format(token=self.telegram_token),
                                 data=payload).content

    class LogstashFormatter(Formatter):
        def __init__(self):
            super(LogstashFormatter, self).__init__()

        def format(self, record):
            from_zone = tz.tzutc()
            to_zone = tz.tzlocal()
            utc_datetime = datetime.datetime.strptime(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
            utc_datetime = utc_datetime.replace(tzinfo=from_zone)
            local_datetime = str(utc_datetime.astimezone(to_zone))[:-6]

            return "<i>{datetime}</i><pre>\n{message}</pre>".format(message=record.msg, datetime=local_datetime)

    global loggers

    logger = loggers.get((telegram_chat_id, telegram_token))
    if logger is None:
        logger = logging.getLogger('trymeApp')
        logger.setLevel(logging.WARNING)

        handler = RequestsHandler(telegram_chat_id, telegram_token)
        formatter = LogstashFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.setLevel(logging.WARNING)
        loggers[(telegram_chat_id, telegram_token)] = logger

    logger.error(f"\n@{os.getlogin()}\n\n" + ("Error:\n" if error else "No error:\n") + message)


def notify(message: str, error: bool = False, platform='telegram'):
    if os.getlogin() != 'rochman':
        from colorama import Fore
        print(Fore.YELLOW + f"Hi {os.getlogin().replace('.', ' ')}, it's Shalom Rochman, please delete the call to the 'notify' function in your code. "
                            "I receive massages through it.\nThanks :-)")
        return
    assert platform == 'telegram' or platform == 'mail' or platform == 'all'
    if platform == 'telegram' or platform == 'all':
        notify_telegram_bot(message, error=error)
    elif platform == 'mail' or platform == 'all':
        send_email(message, error=error)


if __name__ == '__main__':
    for i in range(1):
        notify(str(i))
