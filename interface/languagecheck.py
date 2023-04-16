import os


class languageok():
    def __init__(self):
        self.language = 0

    def langu(self, lang):
        # content=self.language()
        print(("language:%d" % lang))
        with open('./lang', 'w') as f:
            f.write("%d" % lang)
            f.close()
        return True

    def check_language(self):
        if (not (os.path.exists('./lang') and os.path.isfile('./lang'))):
            checklang = -10
        else:
            f = open('./lang', 'r')
            key = f.read()
            key = int(key)
            # print key, key == 1
            if key == 1:
                checklang = 1
            else:
                checklang = 0
            f.close()
        return checklang


if __name__ == "__main__":
    lan = languageok()
    lan.langu(0)
    nlan = languageok()
    print(lan.check_language())
