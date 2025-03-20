from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver import Keys, ActionChains
import time

service = Service(executable_path="/Users/lestersanjuan/Desktop/ZottaGo/IloveTyping/chromedriver")
driver = webdriver.Chrome(service=service)

driver.implicitly_wait(2)

driver.get("https://monkeytype.com/")

driver.implicitly_wait(2)



def time_to_cheat():
    typed_words = driver.find_elements(By.ID, "words")
    about_to_type = driver.find_elements(By.CLASS_NAME, "word")
    print("These are the type words over here=> ", typed_words, "\n about to be typed here =>", about_to_type)

    for w in about_to_type:
        try:
            for s in w.text:
                time.sleep(0.02)
                ActionChains(driver).send_keys(s).perform()
        except:
            print("it breaks lol")
            continue
        ActionChains(driver).send_keys(" ").perform()

        print(w.text)
    
        to_type = driver.find_element(By.CLASS_NAME, "word")
        about_to_type.append(to_type)
    print(about_to_type)

while True:
    input_taker = input("press y whenever youre ready to cheat\n")
    if input_taker == "y":
        try:
            time_to_cheat() 
        except:
            print("hehe lol xD")
    elif input_taker == "quit":
        break
    else:
        print('check')
