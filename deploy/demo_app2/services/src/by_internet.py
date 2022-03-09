import requests

class ChatRobotByInternet():
    def __init__(self, url_prefix):
        # self.url_prefix = "https://api.ownthink.com/bot?appid=xiaosi&userid=user&spoken="
        self.url_prefix = url_prefix

    def answer(self, question):
        url = self.url_prefix + question
        try:
            data = requests.post(url).json()
            if "message" in data and data["message"] == "success":
                return data["data"]["info"]["text"]
            else:
                return None
        except:
            return None


if __name__ == "__main__":
    robot = ChatRobotByInternet()
    while 1:
        q = input()
        if q == 'q':
            break
        answer = robot.answer(q)
        if answer is not None:
            print(answer)
        else:
            print("你说啥？没听懂欸。")