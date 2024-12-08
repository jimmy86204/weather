# 訓練程式

本文件說明如何完成模型訓練與預測的完整流程。

## 步驟說明

1. **準備資料**  
   - 預先下載主辦方提供的訓練、測試及補充資料，並放置於 `data/` 目錄底下。
   - 從[政府開放資料](https://data.gov.tw/dataset/33029)下載政府提供的開放資料：
     1. 進入頁面後，點選資料資源下載網址欄的 **JSON** 按鈕。
     2. 將下載的檔案命名為 `extra.json`，並放置於 `data/` 目錄底下。

2. **安裝依賴套件**  
   要確保所有程式可以正常執行，請先安裝所需的依賴套件。

   #### **安裝方法**：
   1. 打開終端機或命令列工具，執行以下指令：
      ```bash
      pip install -r requirements.txt
      ```

   2. 如果您的電腦配有 GPU ，並希望使用 GPU 加速，請參考 [PyTorch 官方文件](https://pytorch.org/get-started/locally/) 安裝對應的 GPU 支援版本。

   3. 預設環境：
      - 本程式基於 **CPU 環境** 測試，建議在沒有 GPU 的情況下直接安裝預設的依賴套件。
      - 如果欲使用 GPU 加速訓練，請務必確保 CUDA 驅動及相關版本相容。

3. **處理額外資料**  
   執行以下指令以處理 `extra.json`，萃取氣候相關資訊並結構化為 Parquet 格式檔案：
   ```bash
   python 0.process_extra_data.py
   ```
   - **輸出結果**：`data/open_weather_data.parq`

4. **訓練 Model 1**  
   執行以下指令訓練 Model 1，並生成預測結果與 OOF（Out-Of-Fold）資料：
   ```bash
   python 1.training_model1.py
   ```
   - **輸出結果**：儲存於 `output/model1/`

5. **訓練 Model 2**  
   執行以下指令訓練 Model 2，並生成預測結果與 OOF 資料：
   ```bash
   python 2.training_model2.py
   ```
   - **輸出結果**：儲存於 `output/model2/`

6. **模型集成（Ensemble）**  
   執行以下指令進行模型集成，計算最佳組合分數並生成最終的預測結果：
   ```bash
   python 3.ensemble.py
   ```
   - **輸出結果**：最終集成結果儲存於 `output/sub.csv`

## 資料結構

- `data/`  
  儲存原始資料與處理後的中間檔案：
  - `open_weather_data.parq`：由 `0.process_extra_data.py` 生成的結構化檔案。

- `output/`  
  - `model1/`：儲存 Model 1 的預測結果與 OOF 資料。
  - `model2/`：儲存 Model 2 的預測結果與 OOF 資料。
  - `sub.csv`：最終集成後的預測結果。
