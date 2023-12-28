import os
import xml.etree.ElementTree as ET


xml_folder = './data/extract_img_test/extracted_img_231116_인덱싱에러수정(1)/25024694_현윤옥_(1)_YP_인덱싱수정'
yolo_txt_dir = './data/yolo_label_test/yolo_output_231120'


def get_class_index(label):
    '''클래스 레이블을 클래스 인덱스로 매핑'''

    class_mapping = {'Hyoid_Bone': 0,
                     'Neck_Bone': 1, 'Food_Locate': 2, 'Food': 3}
    return class_mapping.get(label, 3)  # Default to 3 if label not found


def convert_to_YOLOformat(xml_folder, yolo_txt_dir):
    '''xml 파일에서 클래스별 바운딩박스 좌표를 YOLO 포맷으로 변환(레이블 데이터 가공)하는 함수'''

    if not os.path.exists(yolo_txt_dir):
        os.makedirs(yolo_txt_dir)

    for file in os.listdir(xml_folder):
        if file.endswith('.xml'):
            print(file)
            tree = ET.parse(os.path.join(xml_folder, file))
            root = tree.getroot()

            for image in root.iter('image'):
                labels = ''  # 레이블 초기화
                image_name = image.attrib['name']
                label_name = image_name.replace('.jpg', '.txt')

                width = float(image.attrib['width'])
                height = float(image.attrib['height'])

                for box in image.iter('box'):
                    label = box.attrib['label']
                    cls = get_class_index(label)

                    xtl = float(box.attrib['xtl'])
                    ytl = float(box.attrib['ytl'])
                    xbr = float(box.attrib['xbr'])
                    ybr = float(box.attrib['ybr'])

                    x_center = (xtl + (xbr - xtl) / 2) / width
                    y_center = (ytl + (ybr - ytl) / 2) / height
                    box_w = (xbr - xtl) / width
                    box_h = (ybr - ytl) / height

                    labels += f"{cls} {x_center} {y_center} {box_w} {box_h}\n"

                with open(os.path.join(yolo_txt_dir, label_name), 'w') as f:
                    f.write(labels)
