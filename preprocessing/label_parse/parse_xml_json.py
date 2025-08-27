import os
import xmltodict
import logging

# Logging Setting 로깅 설정
logging.basicConfig(filename='conversion_log.txt', level=logging.DEBUG)

# variable/config setting 변수 설정
CLASS_MAPPING = {'Hyoid_Bone': 0,
                 'Neck_Bone': 1, 'Food_Locate': 2, 'Food': 3}
parent_dir = '/home/sh_rsrehab/AKAS_Test/data/extracted_img_231116_인덱싱에러수정(1)/'
output_folder = './data//yolo_label_test/labels_231204/'


def get_class_index(label):
    '''클래스 레이블을 클래스 인덱스로 매핑'''
    return CLASS_MAPPING .get(label, 3)  # Default to 3 if label not found


def parse_xml_file(file_path):
    with open(file_path, "r") as f:
        return xmltodict.parse(f.read())


def process_img_data(img_data, output_folder):

    img_name = img_data['@name']
    img_width = float(img_data['@width'])
    img_height = float(img_data['@height'])

    # 이미지 파일의 이름에서 확장자를 제거하여 txt 파일의 이름을 생성
    txt_file_name = os.path.splitext(img_name)[0] + '.txt'
    txt_file_path = os.path.join(output_folder, txt_file_name)

    with open(txt_file_path, 'w') as file:
        for obj in img_data['box']:
            write_obj_data(obj, img_width, img_height, file)


def process_directory(parent_dir, output_folder):

    for subdir in os.listdir(parent_dir):
        xml_directory = os.path.join(parent_dir, subdir)
        if os.path.isdir(xml_directory):
            process_xml_files(xml_directory, output_folder)


def process_xml_files(xml_dir, output_folder):

    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            data = parse_xml_file(os.path.join(xml_dir, filename))
            xml_img_data = data['annotations']['image']

            for img_data in xml_img_data:
                process_img_data(img_data, output_folder)


def write_obj_data(obj, img_width, img_height, file):

    class_id = get_class_index(obj['@label'])
    if class_id is not None:
        xtl = float(obj['@xtl'])
        ytl = float(obj['@ytl'])
        xbr = float(obj['@xbr'])
        ybr = float(obj['@ybr'])

        x_center = (xtl + (xbr - xtl) / 2) / img_width
        y_center = (ytl + (ybr - ytl) / 2) / img_height
        box_w = (xbr - xtl) / img_width
        box_h = (ybr - ytl) / img_height

        file.write(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")


if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    process_directory(parent_dir, output_folder)
