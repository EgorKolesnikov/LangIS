# coding=utf-8


class Register(object):
    AB_ENGLISH = 'ab_English'
    AB_FRENCH = 'ab_French'
    AB_GERMAN = 'ab_German'
    AB_HEBREW = 'ab_Hebrew'
    AB_PORTUGUESE = 'ab_Portuguese'
    AB_RUSSIAN = 'ab_Russian'
    AB_SPANISH = 'ab_Spanish'
    AHMARIC = 'ahmaric'
    CHINESE = 'chinese'
    JAVANESE = 'javanese'
    KHMER = 'khmer'
    NEPALI = 'nepali'
    SA_AFRIKAANS = 'sa_afrikaans'
    SA_ISIXHOSA = 'sa_isiXhosa'
    SA_SESOTHO = 'sa_sesotho'
    SA_SETSWANA = 'sa_setswana'
    SINHALA = 'sinhala'
    SUNDANESE = 'sundanese'
    UYGHUR = 'uyghur'

    @classmethod
    def all(cls):
        values = []

        for attr in dir(cls):
            if attr.startswith('__'):
                continue
            if callable(getattr(cls, attr)):
                continue
            values.append(getattr(cls, attr))

        return values
