# LPD-Yolo26-Parseq-FFT
本專案首先使用 YOLO26 Pose 偵測車牌位置與四個角點，接著依據角點資訊對車牌影像進行透視變換（projection / rectification），並使用 PARSeq 對投影後的車牌影像進行文字辨識。為了進一步評估偵測與校正品質，系統採用 FFT 分析車牌影像的頻域特徵，據此篩選出歪斜、變形或低品質的車牌樣本。對於判定為歪斜的車牌，系統進一步估計較為正確的角點位置，重新進行幾何校正，使歪斜車牌恢復為較正的視角。最後，再將這些經校正後的車牌重新標記，產生新的訓練資料，用以擴充資料集並持續提升後續車牌偵測與辨識模型的準確度與穩定性。

| 檔名  | 功能 |
| ------------- | ------------- |
| 2lp_skew_k_driven_fft_angle.ipynb | 這個 Notebook 是 FFT 角度驅動的車牌 quad 校正實驗場，核心做法是對每張車牌影像呼叫 estCorrect2D() 取得水平與垂直傾斜角（hOffsetDegree/vOffsetDegree），再依角度正負方向沿四角點的邊線每次移動邊長的 2%，迭代到兩個角度都收斂至 0.001 以下為止，並搭配 PARSeq OCR 在每輪後確認首尾字信心分數；每輪均可視覺化顯示 quad 修正前後與投影車牌對比，最終批次處理所有 selected plates 並將迭代記錄存為 CSV，是 lp_batch_process.py 的前身原型與演算法驗證工具。 |
| lp_batch_process.py  | 是一套車牌批次校正處理器，核心任務是從 results.json 中讀取車牌記錄，對每張車牌影像進行幾何傾斜校正，再透過 PARSeq OCR 模型重新辨識文字，最後將結果寫回 JSON 並產生互動式 HTML 檢視報告。幾何校正的核心是 FFT 偏角分析。程式先對車牌影像做去噪與對比增強，再將 2D 傅立葉頻譜轉換為極座標，找出能量最強的方向來估算水平與垂直方向的傾斜角度，接著用仿射變換迭代修正，直到角度收斂或信心分數足夠高為止。校正完成後，程式呼叫延遲初始化的 PARSeq 模型對校正影像做 OCR，記錄首末字元的信心分數，並將 80×320 的校正縮圖以 Base64 格式嵌入 JSON。在工程層面，程式支援分批執行（--max）與定期備份（--checkpoint），並能自動跳過已處理的節點，方便從斷點續跑。全部處理完成後，會產生一份按垂直傾斜角度絕對值降序排列的分頁 HTML 報告，讓使用者點擊每筆記錄，直觀比對校正前後的車牌影像與各項量化指標。|
| lp_gen_html.py | lp_gen_html.py 是專門負責產生互動式 HTML 檢視器的獨立腳本，它從 results.json 讀取已有 corrected_ocr 欄位的車牌記錄，按垂直傾斜角度絕對值降序排列後，產生一份以 10 列網格分頁顯示的瀏覽介面。每張卡片並排顯示校正前後的縮圖（圖片以外部檔案方式引用 plate_images/，而非 Base64 內嵌，所以 HTML 比 lp_batch_process.py 輕量許多），點擊可開啟 Modal 比對角度、OCR 信心分數，以及 OCR 結果與原始檔名車牌號碼是否相符。此外還內建篩選條件（OCR 字元數、信心分數、角度範圍等）、逐筆勾選功能，以及將勾選結果匯出為 JSON 的 Export 按鈕，方便下游挑選優質樣本使用。  |
| lp_selected_viewer.py | lp_selected_viewer.py 是 lp_gen_html.py 的精簡版，專門針對已從大資料集中挑選出來的 selected_plates_*.json 產生獨立的 HTML 檢視器，功能與介面幾乎相同（分頁網格、篩選條件、勾選匯出、Modal 比對），差異在於輸入來源直接是已篩選的小型 JSON 而非完整 results.json，HTML 檔名也會自動沿用輸入 JSON 的檔名（如 selected_plates_429_viewer.html），方便對不同批次的挑選結果各自產生獨立的檢視報告。 |
| lp_yolo_label_gen.py | lp_yolo_label_gen.py 負責將挑選好的車牌記錄轉換成 YOLOv8-Pose 訓練標記，它從 selected_plates.json 取得目標 plate ID，再到 results.json 查找對應的四角點（優先使用 corrected_quad，也可退回原始 keypoints），將每個四角點正規化為 YOLO Pose 格式（class cx cy w h + 四個 keypoint 各含 x y visibility）並寫入 .txt 標記檔，同時產生 dataset.yaml 供 YOLOv8 直接讀取，最後輸出一份 Canvas-based HTML 視覺化檢視器，可在原圖上疊加原始 keypoints（紅）、校正後 quad（綠）及 bounding box（黃），逐張確認標記品質。python lp_yolo_label_gen.py --selected selected_plates_429.json --results results.json   |
| lp_build_dataset.py | lp_build_dataset.py 負責將挑選好的車牌記錄打包成符合 YOLOv8 標準目錄結構的訓練資料集，它依原圖為單位將資料隨機分割為 train/val 兩份（預設 8:2），對每張圖複製原圖並產生對應的 YOLO Pose 標記檔（優先使用 corrected_quad，找不到才退回原始 keypoints），最後輸出 dataset.yaml，可直接用於 YOLOv8-Pose 訓練，是整條流程的最後一哩路。python lp_build_dataset.py   --json selected_plates_429_full.json   --output-dir ./lp_dataset/ |
| lp_merge_dataset.py | lp_merge_dataset.py 負責將自建的車牌標記資料合併進現有的 Roboflow YOLOv8-Pose zip 包，流程是先解壓原始 zip（保留 train/valid/test 三個 split），再從 selected_plates.json 產生 YOLO Pose 標記並將原圖以 lp_ 前綴複製進指定 split（預設 train），最後更新 data.yaml 並重新打包成新的 zip，讓自建資料與 Roboflow 資料無縫合併，可直接上傳 SageMaker 或其他訓練平台使用。lp_merge_dataset.py     --zip lp-det-v3-job3.v1i.yolov8.zip     --json selected_plates_429_full.json     --output merged_lp_dataset.zip |
| lp-det-v3-job3.v1i.yolov8.zip | 網路標記車牌數據集 https://drive.google.com/file/d/15U10ASuJn-0OrVWmr54RyfAacejtqcrI/view?usp=sharing |
| merged_lp_dataset.zip | 合併後訓練數據集 https://drive.google.com/file/d/1gznqqmOxmbCJx595fmQ_IFDE60hEz3RK/view?usp=sharing |
| lp_review_dataset.py | 這支腳本是一個三步驟的 YOLOv8-Pose 車牌資料集審查與清理工具：第一步用 --zip 解壓資料集後，自動掃描 train/valid/test 每個 split，檢查圖片與 label 是否配對（找出孤兒檔案）、驗證 label 的 17 欄格式（1 class + 4 bbox + 4×(x,y,vis) keypoints）是否正確且數值在 [0,1] 範圍內、bbox 是否過大或過小、四個 keypoint 是否構成凸四邊形（用外積判斷是否交叉）、圖片是否損壞或過小、以及透過 MD5 hash 偵測重複圖片，然後把所有結果（含圖片路徑、bbox、keypoint 座標、issue 清單）嵌入一個深色主題的互動式 HTML 審查頁面，支援 10×10 縮圖格線、分頁、依 split 與 issue 篩選、點擊開啟 modal 顯示 canvas 繪製的 bbox 虛線框與 TL/TR/BR/BL 四角 keypoint 連線、勾選標記要刪除的項目並匯出 delete_list.json 或 clean_list.json；第二步人工在瀏覽器中審查標記後；第三步再用 --delete delete_list.json 從原始 zip 中過濾掉被標記的圖片與 label，重新打包成乾淨的資料集 zip。 python lp_review_dataset.py     --zip merged_lp_dataset.zip     --delete delete_list_29.json     --output cleaned_dataset.zip |
| cleaned_dataset.zip | 乾淨的合併後訓練數據集（訓練用這個） https://drive.google.com/file/d/1TuNa84vkC-3KvU_4fLqxy9RPTWc7dwzu/view?usp=sharing |
| sagemaker_yolo26_cleaned_dataset.ipynb | 這個 Notebook 是完整的 SageMaker 端對端訓練與推論流程，針對 cleaned_dataset.zip（已合併 Roboflow 與自建校正資料）在 ml.g5.2xlarge 上啟動 YOLO26-Pose 訓練 Job，內含自動生成 train.py 與 requirements.txt、非阻塞式啟動 Estimator、從 CloudWatch Logs 即時串流日誌並動態繪製收斂曲線、訓練完成後下載 best.pt 等 artifacts，最後對 ./img 目錄的圖片執行 YOLO-Pose 偵測 + 透視投影 + PARSeq OCR 的完整推論管線，是整個車牌專案從訓練到驗證的總指揮 Notebook。 |
| yolo26-train-1773128236.tgz | sagemaker_yolo26_cleaned_dataset.ipynb 訓練後的結果包  https://drive.google.com/file/d/1YDlkkOo_sQpPTB1ZESYj81XrZiC1QNm7/view?usp=drive_link |






