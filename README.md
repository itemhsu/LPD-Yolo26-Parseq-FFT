# LPD-Yolo26-Parseq-FFT
本專案首先使用 YOLO26 Pose 偵測車牌位置與四個角點，接著依據角點資訊對車牌影像進行透視變換（projection / rectification），並使用 PARSeq 對投影後的車牌影像進行文字辨識。為了進一步評估偵測與校正品質，系統採用 FFT 分析車牌影像的頻域特徵，據此篩選出歪斜、變形或低品質的車牌樣本。對於判定為歪斜的車牌，系統進一步估計較為正確的角點位置，重新進行幾何校正，使歪斜車牌恢復為較正的視角。最後，再將這些經校正後的車牌重新標記，產生新的訓練資料，用以擴充資料集並持續提升後續車牌偵測與辨識模型的準確度與穩定性。

| 檔名  | 功能 |
| ------------- | ------------- |
| lp_batch_process.py  | 是一套車牌批次校正處理器，核心任務是從 results.json 中讀取車牌記錄，對每張車牌影像進行幾何傾斜校正，再透過 PARSeq OCR 模型重新辨識文字，最後將結果寫回 JSON 並產生互動式 HTML 檢視報告。幾何校正的核心是 FFT 偏角分析。程式先對車牌影像做去噪與對比增強，再將 2D 傅立葉頻譜轉換為極座標，找出能量最強的方向來估算水平與垂直方向的傾斜角度，接著用仿射變換迭代修正，直到角度收斂或信心分數足夠高為止。校正完成後，程式呼叫延遲初始化的 PARSeq 模型對校正影像做 OCR，記錄首末字元的信心分數，並將 80×320 的校正縮圖以 Base64 格式嵌入 JSON。在工程層面，程式支援分批執行（--max）與定期備份（--checkpoint），並能自動跳過已處理的節點，方便從斷點續跑。全部處理完成後，會產生一份按垂直傾斜角度絕對值降序排列的分頁 HTML 報告，讓使用者點擊每筆記錄，直觀比對校正前後的車牌影像與各項量化指標。|
| lp_gen_html.py | lp_gen_html.py 是專門負責產生互動式 HTML 檢視器的獨立腳本，它從 results.json 讀取已有 corrected_ocr 欄位的車牌記錄，按垂直傾斜角度絕對值降序排列後，產生一份以 10 列網格分頁顯示的瀏覽介面。每張卡片並排顯示校正前後的縮圖（圖片以外部檔案方式引用 plate_images/，而非 Base64 內嵌，所以 HTML 比 lp_batch_process.py 輕量許多），點擊可開啟 Modal 比對角度、OCR 信心分數，以及 OCR 結果與原始檔名車牌號碼是否相符。此外還內建篩選條件（OCR 字元數、信心分數、角度範圍等）、逐筆勾選功能，以及將勾選結果匯出為 JSON 的 Export 按鈕，方便下游挑選優質樣本使用。  |
| lp_selected_viewer.py | lp_selected_viewer.py 是 lp_gen_html.py 的精簡版，專門針對已從大資料集中挑選出來的 selected_plates_*.json 產生獨立的 HTML 檢視器，功能與介面幾乎相同（分頁網格、篩選條件、勾選匯出、Modal 比對），差異在於輸入來源直接是已篩選的小型 JSON 而非完整 results.json，HTML 檔名也會自動沿用輸入 JSON 的檔名（如 selected_plates_429_viewer.html），方便對不同批次的挑選結果各自產生獨立的檢視報告。 |
| lp_yolo_label_gen.py | lp_yolo_label_gen.py 負責將挑選好的車牌記錄轉換成 YOLOv8-Pose 訓練標記，它從 selected_plates.json 取得目標 plate ID，再到 results.json 查找對應的四角點（優先使用 corrected_quad，也可退回原始 keypoints），將每個四角點正規化為 YOLO Pose 格式（class cx cy w h + 四個 keypoint 各含 x y visibility）並寫入 .txt 標記檔，同時產生 dataset.yaml 供 YOLOv8 直接讀取，最後輸出一份 Canvas-based HTML 視覺化檢視器，可在原圖上疊加原始 keypoints（紅）、校正後 quad（綠）及 bounding box（黃），逐張確認標記品質。python lp_yolo_label_gen.py --selected selected_plates_429.json --results results.json   |






