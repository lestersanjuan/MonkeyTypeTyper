from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

service = Service(executable_path="/Users/lestersanjuan/Desktop/ZottaGo/IloveTyping/chromedriver")
driver = webdriver.Chrome(service=service)

driver.implicitly_wait(2)

driver.get("https://monkeytype.com/")

driver.implicitly_wait(2)

typed_words = driver.find_element(By.ID, "words")
about_to_type = driver.find_element(By.CLASS_NAME, "word")
print("These are the type words over here=> ", typed_words, "\n about to be typed here =>", about_to_type)


time.sleep(30)

