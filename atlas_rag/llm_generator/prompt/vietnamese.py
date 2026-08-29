"""Vietnamese adaptation of the three extraction and concept-induction stages.

Keep the upstream JSON keys unchanged for schema/CSV compatibility. Only the
instructions and generated semantic content are localized.
"""

TRIPLE_VI = {
    "system": "Bạn là trợ lý trích xuất tri thức. Chỉ trả về một mảng JSON hợp lệ, không giải thích, không markdown. "
              "Giữ nguyên tên riêng và dấu tiếng Việt. Nội dung trích xuất phải dựa trên đoạn văn, không tự thêm sự thật. "
              "Giữ nguyên các khóa JSON tiếng Anh được yêu cầu; viết giá trị bằng tiếng Việt.",
    "entity_relation": """Trích xuất các sự thật thực thể-quan hệ được nêu rõ trong đoạn văn.
Head và Tail phải là tên/cụm có căn cứ trực tiếp trong đoạn, không tự tạo thực thể.
Quan hệ phải ngắn gọn nhưng có nghĩa; không dùng riêng các từ nối như "của", "và",
"trong", "tại", "cho" làm quan hệ. Mỗi sự thật chỉ xuất một lần. Dùng tên thực thể
cụ thể, không dùng đại từ thay thực thể. Tối đa 25 triple. Chỉ xuất mảng JSON theo mẫu:
[{"Head": "thực thể đầu", "Relation": "quan hệ", "Tail": "thực thể cuối"}]
Đoạn văn:
""",
    "event_entity": """Trích xuất các sự kiện có hành động, thay đổi hoặc diễn biến được nêu rõ.
Không biến mọi mô tả phân loại đơn giản thành sự kiện. Mỗi sự kiện là một câu độc lập,
đầy đủ và cụ thể; mỗi sự kiện chỉ xuất một lần. Liệt kê duy nhất các thực thể thực sự
tham gia và có trong đoạn văn. Tối đa 12 sự kiện, không dùng dấu ba chấm. Chỉ xuất mảng JSON theo mẫu:
[{"Event": "một câu đơn mô tả sự kiện", "Entity": ["thực thể 1", "thực thể 2"]}]
Đoạn văn:
""",
    "event_relation": """Chỉ trích xuất quan hệ thời gian hoặc nhân quả ĐƯỢC NÓI RÕ giữa hai sự kiện.
Relation phải chính xác là một trong: "trước", "sau", "cùng thời điểm", "bởi vì", "dẫn đến".
Head và Tail phải là hai câu sự kiện đầy đủ, mỗi câu có chủ thể và hành động; không được
chỉ là tên người, địa điểm, ngày tháng, danh từ hay mảnh câu. Không dùng quan hệ phân loại,
sở hữu, tác giả, thành viên hoặc đồng nghĩa. Việc hai câu đứng cạnh nhau hay có hai mốc năm
không tự chứng minh quan hệ thời gian/nhân quả. "Sau khi" không tự có nghĩa là "dẫn đến".
Không suy đoán, không dùng đại từ mơ hồ, không lặp và không nối một sự kiện với chính nó.
Nếu đoạn văn không có quan hệ đáp ứng đầy đủ các điều kiện trên, trả về []. Tối đa 8 quan hệ.
Chỉ xuất mảng JSON theo mẫu:
[{"Head": "câu mô tả sự kiện 1", "Relation": "quan hệ thời gian hoặc nhân quả", "Tail": "câu mô tả sự kiện 2"}]
Đoạn văn:
""",
}

_CONCEPT_RULES = """Đưa ra ít nhất 3 cụm khái niệm ngắn ở các mức trừu tượng khác nhau nếu có thể.
Các cụm phải biểu thị loại hoặc khái niệm liên quan trực tiếp, không lặp tên đầu vào
hay lặp ý. Không tự thêm khái niệm không liên quan để đủ số lượng.
Viết bằng tiếng Việt, chỉ một dòng các cụm phân cách bằng dấu phẩy ASCII, không
JSON, không giải thích. Không dùng dấu phẩy bên trong một cụm.
"""

CONCEPT_VI = {
    "entity": _CONCEPT_RULES + "\nKhái quát hóa THỰC THỂ sau, sử dụng ngữ cảnh láng giềng để xác định loại.\n"
              "THỰC THỂ: [ENTITY]\nNGỮ CẢNH: [CONTEXT]\nCác khái niệm:",
    "event": _CONCEPT_RULES + "\nKhái quát hóa SỰ KIỆN thành các loại sự kiện.\nSỰ KIỆN: [EVENT]\nCác khái niệm:",
    "relation": _CONCEPT_RULES + "\nKhái quát hóa QUAN HỆ thành các loại quan hệ.\nQUAN HỆ: [RELATION]\nCác khái niệm:",
}
