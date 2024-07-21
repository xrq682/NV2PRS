import cv2
import numpy
import pandas as pd
from paddleocr import PaddleOCR
import os
from moviepy.editor import *
from pydub import AudioSegment
from pydub.silence import split_on_silence
from nltk import *
from nltk.corpus import stopwords
import datetime
import fractions
import whisper


#activate paddle_env

## videos_to_datasets:

def get_cap_duration(cap):  # 获取视频时长
    rate = cap.get(5)
    frame_num = cap.get(7)
    duration = frame_num/rate
    if duration > 300:
        duration = 'L'
    else:
        duration = 'S'
    return duration

def get_cap_colour(cap): # 获取视频色彩
    # 记录上一帧的平均色彩，初始值为空
    last_avg_color = None
    
    # 定义色调范围
    dark_range = (0, 100, 0, 100, 0, 100) #昏暗
    bright_range = (150, 255, 150, 255, 150, 255) #明快
    lively_range = (70, 255, 50, 255, 0, 20) #活泼
    soft_range = (180, 255, 140, 255, 130, 255) #柔和
    warm_range = (230, 255, 210, 255, 180, 255) #温暖
    lyrical_range = (140, 200, 80, 255, 20, 255) #抒情
    exciting_range = (150, 200, 100, 255, 40, 255) #激动
    nostalgic_range = (130, 180, 90, 255, 40, 255) #怀旧
    
    # 读取每一帧视频
    while True:
        ret, frame = cap.read()
    
        # 如果没有更多的帧，退出循环
        if not ret:
            break
        # 获取每一帧的平均色彩
        avg_color_per_row = numpy.average(frame, axis=0)
        avg_color = numpy.average(avg_color_per_row, axis=0)
        # 根据视频平均色彩是否在定义的色调范围内来判断视频的整体色调
        if (dark_range[0] <= avg_color[0] <= dark_range[1] and 
            dark_range[2] <= avg_color[1] <= dark_range[3] and 
            dark_range[4] <= avg_color[2] <= dark_range[5]):
            colour='昏暗' # 视频为昏暗色调
            break
        elif (bright_range[0] <= avg_color[0] <= bright_range[1] and 
            bright_range[2] <= avg_color[1] <= bright_range[3] and 
            bright_range[4] <= avg_color[2] <= bright_range[5]):
            colour='明快' # 视频为明快色调
            break
        elif (lively_range[0] <= avg_color[0] <= lively_range[1] and 
            lively_range[2] <= avg_color[1] <= lively_range[3] and 
            lively_range[4] <= avg_color[2] <= lively_range[5] and 
            (last_avg_color is None or 
            lively_range[0] <= last_avg_color[0] <= lively_range[1] and 
            lively_range[2] <= last_avg_color[1] <= lively_range[3] and 
            lively_range[4] <= last_avg_color[2] <= lively_range[5])):
            colour='活泼' # 视频为活泼色调
            break
        elif (soft_range[0] <= avg_color[0] <= soft_range[1] and 
            soft_range[2] <= avg_color[1] <= soft_range[3] and 
            soft_range[4] <= avg_color[2] <= soft_range[5] and 
            (last_avg_color is None or 
            soft_range[0] <= last_avg_color[0] <= soft_range[1] and 
            soft_range[2] <= last_avg_color[1] <= soft_range[3] and 
            soft_range[4] <= last_avg_color[2] <= soft_range[5])):
            colour='柔和' # 视频为柔和色调
            break
        elif (warm_range[0] <= avg_color[0] <= warm_range[1] and 
            warm_range[2] <= avg_color[1] <= warm_range[3] and 
            warm_range[4] <= avg_color[2] <= warm_range[5] and 
            (last_avg_color is None or 
            warm_range[0] <= last_avg_color[0] <= warm_range[1] and 
            warm_range[2] <= last_avg_color[1] <= warm_range[3] and 
            warm_range[4] <= last_avg_color[2] <= warm_range[5])):
            colour='温暖' # 视频为温暖色调
            break
        elif (lyrical_range[0] <= avg_color[0] <= lyrical_range[1] and 
            lyrical_range[2] <= avg_color[1] <= lyrical_range[3] and 
            lyrical_range[4] <= avg_color[2] <= lyrical_range[5] and 
            (last_avg_color is None or 
            lyrical_range[0] <= last_avg_color[0] <= lyrical_range[1] and 
            lyrical_range[2] <= last_avg_color[1] <= lyrical_range[3] and 
            lyrical_range[4] <= last_avg_color[2] <= lyrical_range[5])):
            colour='抒情' # 视频为抒情色调
            break
        elif (exciting_range[0] <= avg_color[0] <= exciting_range[1] and 
            exciting_range[2] <= avg_color[1] <= exciting_range[3] and 
            exciting_range[4] <= avg_color[2] <= exciting_range[5] and 
            (last_avg_color is None or 
            exciting_range[0] <= last_avg_color[0] <= exciting_range[1] and 
            exciting_range[2] <= last_avg_color[1] <= exciting_range[3] and 
            exciting_range[4] <= last_avg_color[2] <= exciting_range[5])):
            colour='激动' # 视频为激动色调
            break
        elif (nostalgic_range[0] <= avg_color[0] <= nostalgic_range[1] and 
            nostalgic_range[2] <= avg_color[1] <= nostalgic_range[3] and 
            nostalgic_range[4] <= avg_color[2] <= nostalgic_range[5] and 
            (last_avg_color is None or 
            nostalgic_range[0] <= last_avg_color[0] <= nostalgic_range[1] and 
            nostalgic_range[2] <= last_avg_color[1] <= nostalgic_range[3] and 
            nostalgic_range[4] <= last_avg_color[2] <= nostalgic_range[5])):
            colour='怀旧' # 视频为怀旧色调
            break
        else : colour='未知' # 未知色调
        last_avg_color = avg_color
    return colour

def get_cap_object(cap): # 识别主体
    # 初始化 OCR 模型
    ocr = PaddleOCR(lang='en',use_angle_cls = True) 

    # 初始化目标文本列表
    target_words = ['journalist', 'correspondent','reporter','spokesman','spokesperson','president','professor']

    # 定义裁剪区域的坐标
    x, y, w, h = 20, 50, 600, 265

    # 定义跳帧数
    skip_frames = 100

    # 统计目标文本出现的次数
    counts = {word: 0 for word in target_words}
    while True:
        # 读取一帧视频并进行文本识别
        ret, frame = cap.read()
        if not ret:
            break

        if cap.get(cv2.CAP_PROP_POS_FRAMES) % skip_frames == 0:
            # 处理视频帧
            
            # 将帧转换为RGB图像
            roi = frame[y:y+h, x:x+w]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # 对当前帧进行文本识别
            result = ocr.ocr(roi, cls=True)
            
            # 查找对应词
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    if target_words[0] in line[1][0].lower():
                        counts[target_words[0]] += 1
                    elif target_words[1] in line[1][0].lower():
                        counts[target_words[1]] += 1
                    elif target_words[2] in line[1][0].lower():
                        counts[target_words[2]] += 1
                    elif target_words[3] in line[1][0].lower():
                        counts[target_words[3]] += 1
                    elif target_words[4] in line[1][0].lower():
                        counts[target_words[4]] += 1
                    elif target_words[5] in line[1][0].lower():
                        counts[target_words[5]] += 1
                    elif target_words[6] in line[1][0].lower():
                        counts[target_words[6]] += 1

    if counts['journalist']>0 or counts['correspondent']>0 or counts['reporter']>0:
        object='J'
    elif counts['spokesman']>0 or counts['spokesperson']>0 or counts['president']>0 or counts['professor']>0:
        object='E'
    else: object='O'

    return object
    # 读取视频文件

def get_large_audio_transcription(path): # 将音频文件分割成块的函数,应用语音转文字
  
    video = VideoFileClip(path+'.mp4')
    audio = video.audio
    audio.write_audiofile(path+'.wav')
    
    model = whisper.load_model('base')
    result = model.transcribe(path+'.wav')
    text=''
    for segment in result["segments"]:
        text=text+segment["text"]
    print(text)
    # 返回检测到的所有块的文本
    return text

def words_freq_max(text): # 获取文章最高频的词
    cutwords1=word_tokenize(text.lower())
    # 去除标点符号
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\"','``','\'\'','-','']   #定义符号列表
    cutwords2 = [word for word in cutwords1 if word not in interpunctuations]   
    # 去除停用词
    stops = set(stopwords.words("english"))
    stops2 = ['none','\'s','say','said','says','us','also','go','went','take','men','since','n\'t','\'m','\'ve','\'ll','\'','\'re','un']
    cutwords3 = [word for word in cutwords2 if word not in stops]
    cutwords3 = [word for word in cutwords3 if word not in stops2]
    freq = FreqDist(cutwords3)
    return freq.max()

def determine_tense(sentence): # 获取句子时态
    # 对句子进行词汇分割
    tokens = word_tokenize(sentence)

    # 标识词汇的词性
    tagged_tokens = pos_tag(tokens)

    # 初始化时态为未知
    tense = '0'

    # 判断词性标记，确定时态
    for i in range(len(tagged_tokens)-1, -1, -1):
        if tagged_tokens[i][1] == 'VBD': # VBD：过去式动词
            tense = '过去时' # 过去时
            break
        elif tagged_tokens[i][1] == 'VBP' or tagged_tokens[i][1] == 'VBZ': #VBP：现在时第一人称单数动词；VBZ：现在时第三人称单数动词
            tense = '现在时' # 现在时
            break
        elif tagged_tokens[i][1] == 'VBG': # VBG：现在分词动词
            tense = '现在进行时' # 现在进行时
            break
        elif tagged_tokens[i][1] == 'VBN': # VBN：过去分词动词
            tense = '过去分词' # 过去分词
            break
        elif tagged_tokens[i][1] == 'MD': # MD：情态动词
            tense = '将来时' # 将来时
            break

    return tense

def get_most_frequent_tense(text): # 获取文章句子最多的时态
    
    # 对文章进行句子分割
    sentences = sent_tokenize(text)

    # 统计出现次数最多的时态
    tense_counts = {}
    for sentence in sentences:
        tense = determine_tense(sentence)
        if tense in tense_counts:
            tense_counts[tense] += 1
        else:
            tense_counts[tense] = 1
    if '0' in tense_counts:
        tense_counts['0']=0
    most_frequent_tense = max(tense_counts, key=tense_counts.get)
    return most_frequent_tense

def save_to_excel_sheet(file_path,sheet_name,df): # 将数据框存入指定表格
    
    # 判断文件是否存在，如果不存在则创建新文件
    if not os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


## datasets_to_script:


def days(sheet_first,sheet_last): # 计算投放天数
    
    #该阶段投放的第一天
    day1=datetime.datetime.strptime(sheet_first['日期'].min(),'%Y.%m.%d') 
    #该阶段投放的最后一天
    day2=datetime.datetime.strptime(sheet_last['日期'].max(),'%Y.%m.%d') 
    #获取投放日期天数
    day3=day2-day1 
    day=day3.days+1
    
    return day


def stage(sheet,df,sheet_name,day,day_all): # 各阶段内容处理

    
    date_counts=sheet['日期'].value_counts()
    num_max=date_counts.max() # 每日最大视频投放量
    num_min=date_counts.min() # 每日最小视频投放量

    # 求该阶段L、S、J、E、O的数量

    sheet_L=int((sheet['视频长度'] == 'L').sum())
    sheet_S=int((sheet['视频长度'] == 'S').sum())
    sheet_J=int((sheet['视频主体'] == 'J').sum())
    sheet_E=int((sheet['视频主体'] == 'E').sum())
    sheet_O=int((sheet['视频主体'] == 'O').sum())
    
    colour_counts=sheet['视频色调'].value_counts()
    #del colour_counts['未知']
    colour_max=colour_counts.idxmax() # 出现频次最高的视频色调

    sense_counts=sheet['视频最常用时态'].value_counts()
    sense_max=sense_counts.idxmax() # 出现频次最高的时态

    freq_words = sheet['视频最高频词'].unique()
    freq_words_str = ','.join(freq_words) # 出现的高频词

    #求每天长短视频投放规律
    gl1=''
    if sheet_L==0 :
        gl1='只投放短视频'
    elif sheet_L==0 :
        gl1='只投放长视频'
    elif sheet_L>sheet_S :
        if sheet_L/sheet_S>=2 :
            gl1='同时投放长视频与短视频,且以投放长视频为主'
        else :
            gl1='同时投放长视频与短视频'
    elif sheet_L<sheet_S :
        if sheet_S/sheet_L>=2 :
            gl1='同时投放长视频与短视频,且以投放短视频为主'
        else :
            gl1='同时投放长视频与短视频'
    else:
        gl1='同时投放长视频与短视频，且投放比例相同'
    
    #求叙事主体投放规律
    gl2=''
    if sheet_E==0 and sheet_O==0 :
        gl2='只连线记者'
    elif sheet_J==0 and sheet_O==0 :
        gl2='只连线专家/官员'
    elif sheet_J==0 and sheet_E==0 :
        gl2='只采访平民'
    
    elif sheet_O==0 :
        if sheet_J / sheet_E>=2:
            gl2='连线记者与专家/官员，且以联系记者为主'
        elif sheet_E / sheet_J>=2:
            gl2='连线记者与专家/官员，且以联系专家/官员为主'  
        else: gl2='连线记者与专家/官员'
    elif sheet_E==0 :
        if sheet_J / sheet_O>=2:
            gl2='连线记者与平民，且以联系记者为主'
        elif sheet_O / sheet_J>=2:
            gl2='连线记者与平民，且以联系平民为主'  
        else: gl2='连线记者与平民'
    elif sheet_J==0 :
        if sheet_E / sheet_O>=2:
            gl2='连线专家/官员与平民，且以联系专家/官员为主'
        elif sheet_O / sheet_E>=2:
            gl2='连线专家/官员与平民，且以联系平民为主'  
        else: gl2='连线专家/官员与平民'   
    else:
        if sheet_J / sheet_E>=2 and sheet_J / sheet_O>=1.5  :
            gl2='均连线,记者为主'
        elif sheet_E / sheet_J>=2 and sheet_E / sheet_O>=1.5  :
            gl2='均连线,专家/官员为主'
        elif sheet_O / sheet_J>=2 and sheet_O / sheet_E>=1.5  :
            gl2='均连线,平民为主'
        else:gl2='均连线'

    df_append = [sheet_name,str(day)+'/'+str(day_all),str(num_min)+'~'+str(num_max),gl1,str(fractions.Fraction(sheet_L,sheet_S)),gl2,str(sheet_J)+':'+str(sheet_E)+':'+str(sheet_O),colour_max,sense_max,freq_words_str]
    df.loc[len(df)]=df_append
    
    return df


def to_script(dataset_path): # 生成脚本模板
     
    #导入数据
    sheet=pd.read_excel(dataset_path,sheet_name=None)
    
    #建立脚本的空数据框
    df = pd.DataFrame([], columns=['阶段','投放日比例参考','数目个/天','每天长短视频投放规律','长/短','每天投放规律','记者：专家：平民','视频色调','视频时态','高频词']) 
    
    #计算整体投放天数
    sheet_first=pd.read_excel(dataset_path,sheet_name=list(sheet.keys())[0])
    sheet_last=pd.read_excel(dataset_path,sheet_name=list(sheet.keys())[-1])
    day_all=days(sheet_first,sheet_last)
 
    #循环各阶段的sheet，得到完整脚本数据框
    for i in list(sheet.keys()):
        
        sheet=pd.read_excel(dataset_path, sheet_name=i)
        day=days(sheet,sheet)
        df=stage(sheet,df,i,day,day_all)
     
    #保存脚本，输出为xlsx格式
    outputpath='D:/x/sys/2023.3.21/video_to_script_new3.26/输出结果/脚本3.26.xlsx'
    df.to_excel(outputpath,index=False,header=True)
    

if __name__ == "__main__": 

    video_dir_all=['预热','结尾']
    path="D:/x/sys/2023.3.6/3.6-3.12周工作/video to script/输入/video"
    #r = sr.Recognizer() # 创建一个语音识别对象

    for video_dir in video_dir_all:
        #print(video_dir)
        results=[]
        video_dir_path=os.path.join(path, video_dir)
        print('video_dir_path:')
        print(video_dir_path)
        for filename in os.listdir(video_dir_path):
            if not filename.endswith('.mp4'):
                continue    
            # 读取视频
            video_path = os.path.join(video_dir_path, filename)    
            video = cv2.VideoCapture(video_path)
            text_path=video_path.split(".mp4")[0]
            video_name=filename.split(".mp4")[0]
            text=get_large_audio_transcription(text_path) 
            results.append([video_name,get_cap_duration(video),get_cap_colour(video),get_cap_object(video),words_freq_max(text),get_most_frequent_tense(text)])
            os.remove(text_path+'.wav')
            video.release()
          
        results_df = pd.DataFrame(results, columns=['视频', '视频长度', '视频色调','视频主体','视频最高频词','视频最常用时态'])
        spxx=pd.read_excel("D:/x/sys/2023.3.21/video_to_script_new3.26/输入/视频信息库.xlsx")
        spxx=spxx[['视频','日期']]
        hb=pd.merge(spxx,results_df,on='视频') # 存入日期

        dataset_path = 'D:/x/sys/2023.3.21/video_to_script_new3.26/输出结果/风格库例3.26.xlsx' # 设置存入的excel地址
        sheet_name = video_dir # 设置存入的sheet名
        save_to_excel_sheet(dataset_path,sheet_name,hb) # 存入风格库excel
     
    to_script(dataset_path)