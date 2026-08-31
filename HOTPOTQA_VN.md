# Chạy AutoSchemaKG trên HotpotQA-VN

Nguồn: [Prepare-data-HotpotQA-VN](https://github.com/chichic21039/Prepare-data-HotpotQA-VN),
đọc **`data/hotpotqa_vi_1k/final`**, không dùng `checkpoints/` hoặc tự áp dụng lại review.

Đã kiểm tra nguồn tại commit `b2fde513912b3e17f9a855e759bbdfc2c2970154`:

| File | Nội dung |
| --- | --- |
| `queries.jsonl` | 1.000 câu hỏi, lấy `question_vi` và `answer_vi` |
| `corpus.jsonl` | 9.822 tài liệu, giữ ID nguồn, title và text tiếng Việt |
| `qrels.tsv` | 2.000 nhãn dương; mỗi câu có 2 tài liệu gold |
| `source_qa_issues.csv` | 5 vấn đề nguồn; ghi nhận, không tự sửa/xóa câu |

## Khác biệt quan trọng so với bản tiếng Anh

- `final/` không cung cấp 10 context ứng viên cho từng câu hỏi. Vì vậy script giữ
  **toàn bộ corpus 9.822 tài liệu**, kể cả khi chỉ chọn 3 hoặc 100 câu để đánh giá.
- Không dùng qrels để chỉ giữ tài liệu gold: làm vậy sẽ biến bài toán thành
  truy hồi trên corpus được chọn bằng đáp án. Qrels chỉ nằm trong manifest đánh giá.
- Prompt extraction và concept induction có bản `vi`; không gắn nhãn giả `en`.
  Các khóa JSON và nhãn cấu trúc như `mention in`, `has_concept` vẫn giữ định dạng
  của repo để tương thích bộ chuyển đổi graph.
- Reader dùng tiếng Việt và embedding mặc định là `intfloat/multilingual-e5-small`.
  Encoder thêm `query: ` cho truy vấn và `passage: ` cho tài liệu/triples đúng theo
  [model card E5](https://huggingface.co/intfloat/multilingual-e5-small).
- Retrieval được chấm bằng **corpus ID trong qrels**, không ghép theo tiêu đề
  `supporting_facts` tiếng Anh gốc. Chỉ số câu gốc có thể không còn khớp bản dịch.
- Không dùng graph ZIP tiếng Anh cũ thay cho graph tiếng Việt. Cần dựng graph VN.

## Notebook Colab

File: `colab/AutoSchemaKG_HotpotQA_VN.ipynb`.

[Mở notebook No-Event experiment trên Colab](https://colab.research.google.com/github/phuongth05/SmallScaledAutoSchemaKG/blob/experiment/no-event-small-model/colab/AutoSchemaKG_HotpotQA_VN.ipynb)

Notebook clone cả code lẫn repo dữ liệu, ghim revision, dùng hai môi trường
riêng cho CPU embedding/KG client và GPU vLLM, lưu kết quả vào Drive.
Notebook dùng profile để tách checkpoint. Profile mặc định `ab_entity_event_v3`
chạy pilot ablation 109 chunk với repetition penalty 1.15, giữ entity-relation và
event-entity nhưng không gọi stage event-relation. Kết quả không trộn với baseline,
pilot `ab_rp115` hay `ab_event_guard_v2`.

1. `prepare` có thể chạy CPU. Chọn GPU trước khi chuyển sang `extract`, `build` hoặc `benchmark`.
2. Giữ `RUN_PHASE='prepare'` ở lần đầu để kiểm tra dữ liệu. Bản thân bước prepare
   không cần GPU; nếu đổi loại runtime làm reset `/content`, chạy lại setup với cùng `RUN_ROOT`.
3. Đổi `RUN_PHASE='extract'`, chạy lại cell cấu hình, cell môi trường, cell server và cell chạy phase.
   Extraction checkpoint sau từng chunk trong `graph/kg_extraction/*.json`. Nếu Colab
   disconnect, mở lại cùng `RUN_ROOT` và chạy lại `extract`; wrapper kiểm tra JSONL rồi
   tự bỏ qua các chunk đã hoàn tất. Số chunk mỗi lượt do profile quyết định: pilot
   mặc định dừng sau 109 chunk; baseline dùng 500. Không dùng `--overwrite`.
   Đây là bước tốn tài nguyên trên cả 9.822 tài liệu; chưa có đo đạc thời gian thực
   cho corpus này. Không suy ra thời gian từ việc chỉ chọn 3 câu hỏi.
4. Extraction hoàn tất thì đổi sang `build` để sinh concept và GraphML.
5. Dùng `package` để lưu ZIP graph/provenance, rồi `benchmark` để chạy QA.

Tiến độ resume được ghi ở `graph/extraction_progress.json`. File này chỉ là snapshot;
nguồn sự thật là số JSON record hợp lệ đã flush trong `graph/kg_extraction/*.json`.
JSONL hỏng sẽ làm resume dừng rõ ràng thay vì bỏ qua hoặc tạo graph thiếu dữ liệu.
6. Chạy cell download trước khi disconnect và giữ notebook đã chạy.

`MAX_QUESTIONS` chọn số câu ở **prepare**, không sửa được manifest đã chuẩn bị bằng
cách đổi biến trong bước benchmark. Muốn thay mẫu câu, dùng `RUN_ROOT` mới.
Notebook mặc định không tự chạy toàn bộ extraction/QA chỉ bằng “Run all”.

## CLI theo từng giai đoạn

Clone dữ liệu (chỉ cần một lần):

```bash
git clone https://github.com/chichic21039/Prepare-data-HotpotQA-VN.git /content/Prepare-data-HotpotQA-VN
```

Chạy từ thư mục AutoSchemaKG. Nếu dùng môi trường riêng, thay `python` bằng đường
dẫn interpreter đó. Cài `requirements-hotpotqa-v2.txt` và `requirements-colab.txt`
vào môi trường CPU/client, giữ vLLM trong môi trường riêng như notebook. Khi dùng
uv, chọn `--torch-backend=cpu` cho môi trường client; không cài đè Torch của server.

### 1. Prepare — CPU, không gọi LLM

```bash
python -X utf8 scripts/run_hotpotqa_vn.py \
  --phase prepare \
  --source-dir /content/Prepare-data-HotpotQA-VN/data/hotpotqa_vi_1k/final \
  --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_run \
  --max-questions 1000 --sampling random --seed 42
```

Cũng có thể chỉ dùng bộ chuyển đổi:

```bash
python -X utf8 scripts/prepare_hotpotqa_vn.py \
  --source-dir /content/Prepare-data-HotpotQA-VN/data/hotpotqa_vi_1k/final \
  --output-dir data/hotpotqa_vn --max-questions 1000
```

Không có fallback sang `question_en`/`answer_en` nếu thiếu dữ liệu VI. Script
kiểm tra ID trùng, qrels treo, tài liệu/câu trả lời rỗng; lưu commit nguồn, SHA256
các file và IDs được chọn. Dùng lại cùng input/config thì không ghi đè. Nếu dữ
liệu khác, script yêu cầu thư mục output mới.

### 2. Extraction — cần local Qwen server

Khởi động server như cell vLLM trong notebook; endpoint mặc định là
`http://127.0.0.1:8000/v1`. Script dưới đây **không tự khởi động server**.

```bash
python -X utf8 -u scripts/run_hotpotqa_vn.py \
  --phase extract \
  --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_run \
  --model Qwen/Qwen3.5-2B --base-url http://127.0.0.1:8000/v1
```

### 3. Concepts và GraphML — vẫn cần local Qwen server

```bash
python -X utf8 -u scripts/run_hotpotqa_vn.py \
  --phase build \
  --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_run
```

Dùng cùng model, `--chunk-size`, `--max-new-tokens` và `--model-revision` ở extract
và build nếu đã tùy chỉnh. Chunk mặc định là 3.000 **ký tự**, không phải token.

### 4. Đóng gói và benchmark

```bash
python -X utf8 scripts/run_hotpotqa_vn.py \
  --phase package --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_run

python -X utf8 -u scripts/run_hotpotqa_vn.py \
  --phase benchmark --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_run
```

Benchmark chạy Dense, Entity-KG, Entity-Event-KG và Full-KG trên cùng corpus,
dùng tiếng Việt, E5 CPU và Qwen3.5-2B. Có thể chọn `--variants dense full` hoặc
chạy chẩn đoán `--retrieval-only --no-filter-edges` trong một work/output benchmark
khác; chẩn đoán này không tạo EM/F1 và không phải cấu hình LLM-filter đầy đủ.

Khi đã có ZIP graph VN, có thể dùng trực tiếp runner v2:

```bash
python -X utf8 -u scripts/run_hotpotqa_benchmark.py hotpotqa_vn_graph.zip \
  --output-dir outputs/hotpotqa_vn_benchmark \
  --language vi --embedding-model intfloat/multilingual-e5-small \
  --model Qwen/Qwen3.5-2B --base-url http://127.0.0.1:8000/v1
```

Muốn chạy hết từ đầu bằng một lệnh, dùng `--phase all --source-dir ...`; lệnh này
sẽ chạy cả extraction, concepts và QA, không chỉ prepare.

## Kết quả và resume

Trong `--work-dir`:

```text
input/                    # corpus và manifest đã chuyển đổi; không trộn gold vào text
graph/kg_extraction/      # các lần LLM extraction
graph/kg_graphml/         # graph tiếng Việt
graph/provenance/         # bản sao input + metadata cho ZIP và benchmark
hotpotqa_vn_graph.zip     # tạo bởi package
benchmark/results/       # checkpoint từng câu, evidence, prediction, metrics
benchmark/summary.json   # tiến độ và các điểm trung bình
```

- Prepare và các phase đã đánh dấu hoàn tất được bỏ qua khi chạy lại.
- **Extraction bị ngắt giữa chừng không có automatic resume trong wrapper này.**
  Script dừng nếu thấy output extraction chưa hoàn tất để tránh chạy lặp hoặc xóa
  dữ liệu. Không tự tạo marker complete bằng tay.
- Build bị ngắt có thể chạy lại phase build; bước concept chưa có checkpoint
  từng phần như QA và có thể phải làm lại. Không dùng `--overwrite`.
- Benchmark có checkpoint theo từng câu: khởi động lại server rồi chạy cùng
  lệnh/config để tiếp tục. Thay code/model/dependency/input sẽ cần output mới.
- Không chạy hai tiến trình cùng ghi một work directory. Lưu trên Drive hoặc
  tải ZIP; `/content` đơn thuần sẽ mất khi runtime bị reset.

## Chỉ số VI và giới hạn

EM/F1 VI dùng Unicode NFC, casefold, loại dấu câu, giữ dấu tiếng Việt và không
xóa các từ `a/an/the` theo luật tiếng Anh. F1 tách theo khoảng trắng (đơn vị có thể
là âm tiết tiếng Việt), **không tuyên bố đây là word-segmented Vietnamese F1**.
`yes/no/noanswer` được ánh xạ sang `có/không/không có câu trả lời`; đáp án boolean
sai không được tính điểm giao token một phần. Các tên riêng/đáp án khác không
tự dịch, sửa hay thêm alias.

`support_recall@2/5` và `all_support@2/5` dựa trên qrels ID. Nhiều chunk cùng một
document ID chỉ được tính một lần trong tập truy hồi. Đây là document retrieval
recall, không phải supporting-sentence F1 hoặc joint HotpotQA F1.

Giữ nguyên `answer_vi` đã xuất ở final, kể cả các vấn đề dữ liệu nguồn được
ghi nhận. Không tự áp dụng lại `answer_vi_corrected` từ CSV. Chất lượng dịch và
đáp án nguồn có thể ảnh hưởng điểm; cần phân tích lỗi riêng.

Đây là phần mở rộng tiếng Việt của pipeline thu nhỏ, không phải kết quả tái lập
paper trên benchmark tiếng Anh. Chưa có chạy LLM toàn corpus hay số đo GPU cho
bộ VN này.

## Kiểm thử

## Full vs No-Event pilot

`--without-event-relations` is **not** a no-event mode: it still runs Stage 2
Event--Entity extraction and creates event nodes.  For the actual no-event arm,
use `--without-events`, which retains only Entity--Entity extraction and concept
induction.  Use two new work directories; their construction outputs are not
interchangeable.

Run these phases for each arm with the same prepared input, model revision,
sampling seed and `--max-extraction-chunks 30` (increase only after the pilot):

```bash
# Prepare the same 20--50-query pilot manifest in each new work directory.
python -X utf8 scripts/run_hotpotqa_vn.py --phase prepare --source-dir /content/Prepare-data-HotpotQA-VN/data/hotpotqa_vi_1k/final --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_full_pilot --max-questions 30 --sampling random --seed 42
python -X utf8 scripts/run_hotpotqa_vn.py --phase prepare --source-dir /content/Prepare-data-HotpotQA-VN/data/hotpotqa_vi_1k/final --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_no_event_pilot --max-questions 30 --sampling random --seed 42

# Full: Entity + Event + Concept
python -X utf8 -u scripts/run_hotpotqa_vn.py --phase extract --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_full_pilot --model Qwen/Qwen3.5-2B --base-url http://127.0.0.1:8000/v1 --max-extraction-chunks 30
python -X utf8 -u scripts/run_hotpotqa_vn.py --phase build --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_full_pilot --model Qwen/Qwen3.5-2B --base-url http://127.0.0.1:8000/v1
python -X utf8 -u scripts/run_hotpotqa_vn.py --phase benchmark --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_full_pilot --model Qwen/Qwen3.5-2B --base-url http://127.0.0.1:8000/v1 --variants full --top-passages 10

# No-Event: Entity + Concept (no Event--Entity or Event--Event calls/nodes/edges)
python -X utf8 -u scripts/run_hotpotqa_vn.py --phase extract --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_no_event_pilot --model Qwen/Qwen3.5-2B --base-url http://127.0.0.1:8000/v1 --max-extraction-chunks 30 --without-events
python -X utf8 -u scripts/run_hotpotqa_vn.py --phase build --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_no_event_pilot --model Qwen/Qwen3.5-2B --base-url http://127.0.0.1:8000/v1 --without-events
python -X utf8 -u scripts/run_hotpotqa_vn.py --phase benchmark --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_no_event_pilot --model Qwen/Qwen3.5-2B --base-url http://127.0.0.1:8000/v1 --variants no_event --top-passages 10

python scripts/report_no_event_ablation.py /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_full_pilot /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_no_event_pilot
```

For the notebook, select `full_pilot`, finish all phases, then select
`no_event_pilot` and repeat.  The phase cell supplies `--without-events`,
`--variants no_event`, and `--top-passages 10` automatically.

```bash
python -X utf8 -m pytest tests/test_hotpotqa_vn.py tests/test_hotpotqa_v2.py -q
```

Test kiểm tra chuyển đổi/schema/qrels, không đưa gold vào corpus, giữ toàn corpus,
Unicode và boolean scoring, prefix E5, đăng ký prompt vi, khả năng đọc bundle VN,
và hồi quy retrieval/checkpoint tiếng Anh. Input nguồn và các output thử nghiệm
không được thêm vào Git.
