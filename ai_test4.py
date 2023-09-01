# -*-coding: Utf-8 -*-
# @File : ai_test4.py
# author: lhc
# Time：2023/8/28 19:41
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import Birch



# 停用词
stopwords = []
with open('stopwords.txt', errors='ignore') as sf:
    for line in sf.readlines():
        stopwords.append(line.strip())


def text_cut(in_text):
    words = jieba.lcut(in_text)
    # words2 = jieba.cut(in_text,)
    # cut_text = ' '.join([w for w in words if w not in stopwords and len(w) > 1])
    cut_text = ' '.join([w for w in words if len(w) > 1])
    return cut_text


x = [r'压迫板完成压迫后，应当保持稳定：当压迫板压迫模体，压迫力达到100N时，一分钟内，压迫力变化值不得大于10N.',
     r'系统应支持用户在各种曝光模式下，对相关的曝光参数进行设置。如果是在AEC模式下，只需要按照AEC曝光原则进行相应的设置就可以。',
     r'系统应提供一组满足临床图像质量要求的默认曝光参数。',
     r'辐射野与探测器有效接受面应当尽可能的重合：1 辐射野上、下边缘与探测器接受面上、下边缘的偏差不得大于SID的2%，其中辐射野在胸墙侧应当超过探测器接受面胸墙侧的边缘，但超出部分不能大于5mm。2 辐射野左、右边缘与探测器接受面左、右边缘的偏差不得大于SID的2%。',
     r'28， 输液架： 尺寸： 不小于240*240*870mm， 注意材质： 海绵',
     r'28， 输液架： 尺寸： 不小于240*240*870mm， 主要材质： 金属'
     ]

x_target = [
    r'使用BR3D模体，执行压迫，压迫到100N（系统显示值）后停止压迫，观察1分钟; 压迫力变化不大于10N',
    r'启动系统，注册检查，进入检查页面，选择曝光模式为manual，尝试调整曝光参数;能够调整kv,mAs，滤过/靶面，植入物;切换到AEC曝光模式，尝试调整曝光参数;能够选择uSTD,uDose,uHD，植入物',
    r'登录XPET，遍历所有协议，检查协议参数与APR文档(80000024-SFS-APR/88000014-SFS-APR)是否一致;系统默认参数应与临床发布APR参数表保持一致',
    r'使用DXR检测尺和十字铅尺放置在压迫台上表面，0刻度线与胸墙侧限束器光野平行，将限束器打到最大，采集图像;正常采集图像;查看图像上显示的刻度值，并与探头上显示的辐射野进行对比;胸墙侧辐射野应超出探测器接受面，超出范围误差应小于等于5mm;按照步骤2、3的方法分别测试其他3个边;辐射野胸墙侧和对侧偏差的绝对值之和以及辐射野左侧和右侧偏差的绝对值之和均不超过14mm',
]


if __name__ == '__main__':
    # data = pd.read_excel('文本数据集.xlsx')
    # x = data.留言主题

    x_change = []
    for i in x:
        x_change.append(text_cut(i))

    vectorized = TfidfVectorizer(min_df=2, ngram_range=(1, 2), strip_accents='unicode', norm='l2',
                                 token_pattern=r"(?u)\b\w+\b")
    X = vectorized.fit_transform(x_change)
    num_clusters = 390
    birch_cluster = Birch(n_clusters=num_clusters)
    birch_result = birch_cluster.fit_predict(X)
    print("Predicting result: ", birch_result)

    # data['类别编号'] = birch_result
    # pd.DataFrame(data).to_excel('文本数据集.xlsx', sheet_name='Sheet1', index=False, header=True)
    # pass
    # https://blog.csdn.net/weixin_44613063/article/details/105971614
