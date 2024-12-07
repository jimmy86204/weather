# 訓練程式

1. 預先下載主辦方提供的訓練、測試及補充資料，放在data/目錄底下。
2. 預先從https://data.gov.tw/dataset/33029下載政府開放資料。進入頁面後，點選資料資源下載網址欄的JSON按鈕。下載好後將檔名命名為extra.json，並放在data/目錄底下。
3. 執行0. process_extra_data.py。萃取extra.json內有關氣候的資訊，並且結構化後存在data/目錄底下，檔名為open_weather_data.parq。
4. 執行1. training_model1.py。執行訓練model1，並且得到預測結果及OOF，存放在output/model1/的資料夾裡。
5. 執行2. training_model2.py。blabalabla。
6. 執行3. ensemble.py。計算兩個model最好的組合分數，並產生最終預測結果在路徑output/sub.csv。