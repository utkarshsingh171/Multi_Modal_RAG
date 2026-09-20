import fitz
from PIL import Image
import io,os

pdf_path = "sample.pdf"

doc = fitz.open(pdf_path)

texts = []
images = []

for page_no,page in enumerate(doc):
    # extract text
    text = page.get_text()

    if text.strip():
        texts.append({
            "page":page_no+1,
            "text":text
        })
    # extract images
    for img_no, img in enumerate(page.get_images(full=True)):
        xref = img[0]

        base_image = doc.extract_image(xref)

        image_byte = base_image["image"]

        image = Image.open(io.BytesIO(image_byte))

        images.append({
            "page":page_no+1,
            "image_no":img_no+1,
            "image":image
        })

os.makedirs("extracted_images", exist_ok=True)

for i, item in enumerate(images):

    path = f"extracted_images/image_{i+1}.png"

    item["image"].save(path)

    print(
        f"Page {item['page']} -> {path}"
    )
# print("pages with text:", len(texts))
# print("images found:" ,len(images))


