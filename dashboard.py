import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import openai
import time
import numpy as np
import requests
from datetime import date, timedelta  
import json,pickle

def exchange():
    today = date.today()
    data = None
    max_attempts = 30  # 最多嘗試30天
    attempts = 0
    
    while data is None and attempts < max_attempts:
        print(f"嘗試獲取 {today} 的匯率資料")
        url = "https://api.finmindtrade.com/api/v4/data"
        token = ""  # 參考登入，獲取金鑰
        headers = {"Authorization": f"Bearer {token}"}
        parameter = {
            "dataset": "TaiwanExchangeRate",
            "data_id": "AUD",
            "start_date": today.strftime("%Y-%m-%d"),
        }
        
        try:
            response = requests.get(url, headers=headers, params=parameter)
            response_data = response.json()
            
            if response_data.get('data') and len(response_data['data']) > 0:
                data = pd.DataFrame(response_data['data'])
                print(f"成功獲取 {today} 的匯率資料")
                break
            else:
                print(f"{today} 無匯率資料，嘗試前一天")
                today = today - timedelta(days=1)
                attempts += 1
                
        except Exception as e:
            print(f"API 請求錯誤: {e}")
            today = today - timedelta(days=1)
            attempts += 1
    
    if data is None:
        print("無法獲取匯率資料")
        return today, 0  # 返回預設值
    return today + timedelta(days=1), data['cash_sell'].values[0]  

exchange_date, exchange_rate = exchange()
st.set_page_config(layout="wide")
# 設定自定義顏色配色
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
px.defaults.color_discrete_sequence = custom_colors

if st.session_state.get('median_data') is None:
    st.session_state.median_data = None 
if st.session_state.get('data') is None:
    st.session_state.data = None
@st.cache_data
def load_data(exchange_rate):
    with open('./data/dataMedian.json', 'r', encoding='utf-8') as f:
        st.session_state.median_data = pd.DataFrame(json.load(f))

    with open('./data/data.pkl', 'rb') as f:
        st.session_state.data = pd.DataFrame(pickle.load(f))
    
    st.session_state.data['Purchase Price(NTD)'] = st.session_state.data['Purchase Price(NTD)']  /20*exchange_rate
    st.session_state.data['UNITS'] = st.session_state.data['UNITS']  /20 * exchange_rate

    st.session_state.median_data['medianEHT(台幣)'] = st.session_state.median_data['medianEHT(台幣)']  /20 * exchange_rate
    st.session_state.median_data['medianADT(台幣)'] = st.session_state.median_data['medianADT(台幣)']  /20 * exchange_rate

load_data(exchange_rate)

######################################################################################################################################################
st.title("澳洲雪梨地區房地產資料")
st.caption(f"📊 匯率擷取日期：{exchange_date.strftime('%Y年%m月%d日')} | 💱 澳幣匯率：1 AUD = {exchange_rate:.2f} TWD")

with st.expander("資料預覽", expanded=False):
    if st.session_state.get('data') is not None:
        st.dataframe(st.session_state.median_data)
    else:
        st.warning("尚未載入資料，請先執行查詢。")
######################################################################################################################################################

s1,s2,empty = st.columns([1,2,2])
with s1:
    region_option = st.session_state.data['Greater Sydney District'].unique().tolist()
    region_option.insert(0, '全區')
    dist = st.selectbox("請選擇地區:", options=region_option)
with s2:
    quarter_option = st.session_state.data['Quarter'].unique().tolist()
    quarter_option.sort(reverse=False)
    quarter = st.select_slider("請選擇時間:", options=quarter_option,value=quarter_option[-1], format_func=lambda x: f"{x[:4]}年第{x[5:]}季"  ,label_visibility="visible"  )

######################################################################################################################################################
col1, col2 = st.columns([2,1])

##################
### 房市趨勢 ###
##################

with col1:
    st.subheader(":blue[🏠房市趨勢]", anchor =False, divider='blue')
        
    with st.container(border=True, height=600):
        tab1, tab2 = st.tabs(["總價 中位數","單價 平均數"])
        with tab1:
            # 繪製組合圖
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 添加柱狀圖 - 交易量
            fig.add_trace(
                go.Bar(
                    x=st.session_state.median_data['YYYYQQ'], 
                    y=st.session_state.median_data['交易量'],
                    name='交易量', 
                    marker_color='#4ECDC4', 
                    opacity=0.6
                ),
                secondary_y=False,
            )

            # 添加折線圖 - medianEHT(台幣)
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.median_data['YYYYQQ'], 
                    y=st.session_state.median_data['medianEHT(台幣)'],
                    mode='lines+markers+text', 
                    name='中位數房價-獨棟住宅',
                    text=st.session_state.median_data['medianEHT(台幣)'].round(1),
                    textposition='top center', 
                    textfont=dict(size=12, color='black'),   
                    line=dict(color='#FF6B6B')
                ),
                secondary_y=True,
            )
            
            # 添加折線圖 - medianADT(台幣)
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.median_data['YYYYQQ'], 
                    y=st.session_state.median_data['medianADT(台幣)'],
                    mode='lines+markers+text', 
                    name='中位數房價-公寓/聯排住宅',
                    text=st.session_state.median_data['medianADT(台幣)'].round(1),
                    textposition='bottom center', 
                    textfont=dict(size=12, color='black'),   
                    line=dict(color='#45B7D1')
                ),
                secondary_y=True,
            )
                
            # 設定y軸標題
            fig.update_yaxes(title_text="交易量", secondary_y=False)
            fig.update_yaxes(title_text="中位數房價(萬台幣)", secondary_y=True)
            
            # 設定x軸標題
            fig.update_xaxes(title_text="季度")
            
            # 設定圖例位置
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            turn_time = st.toggle('切換X軸',value=False)
            df = st.session_state.data[st.session_state.data['Greater Sydney District'] == dist] if dist != '全區' else st.session_state.data 
            df['ym'] = pd.to_datetime(df['ym'], format='%Y%m')

            if turn_time:
                df_summary = df.groupby(['yyyy']).agg(
                    average_units=('UNITS', 'mean'),
                    count=('UNITS', 'count')).reset_index()
                df_summary['average_units'] = df_summary['average_units'].round(2)
            else:
                df_summary = df.groupby(['Quarter']).agg(
                    average_units=('UNITS', 'mean'),
                    count=('UNITS', 'count')).reset_index()
                df_summary['average_units'] = df_summary['average_units'].round(2)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bar chart for count
            fig.add_trace(
                go.Bar(x=df_summary['yyyy'] if turn_time else df_summary['Quarter'],
                        y=df_summary['count'],
                        name='交易數量', marker_color='#4ECDC4', opacity=0.6),
                        secondary_y=False,
            )

            # Add line chart for average_units
            fig.add_trace(
                go.Scatter(x=df_summary['yyyy'] if turn_time else df_summary['Quarter'], 
                            y=df_summary['average_units'],
                            mode='lines+markers+text', name='平均單價',text=df_summary['average_units'],
                            textposition='top center', textfont=dict(size=12, color='black'),   
                            line=dict(color='#FF6B6B')),
                            secondary_y=True,
                )
                
            # Set y-axes titles
            fig.update_yaxes(title_text="平均單價(萬/坪)", secondary_y=False)
            fig.update_yaxes(title_text="交易數量", secondary_y=True)
            
            # Set x-axis title
            fig.update_xaxes(title_text="季度")
            
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

####################
### Zoning圓餅圖  ###
####################
with col2:
    st.subheader(f":blue[🏢{quarter[:4]}年第{quarter[5:]}季 - 建物類別]", anchor =False, divider='blue')
    if dist == '全區':
        df2 = st.session_state.data[st.session_state.data['Quarter'] == quarter]
        df_bar = df2.groupby(['Zoning']).agg(count=('UNITS', 'count')).reset_index()
        df_bar2 = st.session_state.data.groupby(['Zoning','yyyy']).agg(mean=('UNITS', 'mean')).reset_index()
    else:
        df2 = st.session_state.data[(st.session_state.data['Quarter'] == quarter) & (st.session_state.data['Greater Sydney District'] == dist)]
        df_bar = df2.groupby(['Zoning']).agg(count=('UNITS', 'count')).reset_index()
        df_bar2 = st.session_state.data[(st.session_state.data['Greater Sydney District'] == dist)].groupby(['Zoning','yyyy']).agg(mean=('UNITS', 'mean')).reset_index()

    with st.container(border=True, height=600):
        tab1, tab2 = st.tabs(["類別分布", "趨勢分析"])
        with tab1:
            # 計算總交易數量
            total_count = df_bar['count'].sum()

            # 建立固定的分區顏色對應
            zoning_color_map = {
                'R1': '#FF6B6B',
                'R2': '#45B7D1', 
                'R3': '#FFA07A',
                'R4': '#F7DC6F',
                'R5': '#BB8FCE'
            }

            # 設定分區順序並重新排序數據
            zoning_order = ['R1', 'R2', 'R3', 'R4', 'R5']
            df_bar_sorted = df_bar.set_index('Zoning').reindex(zoning_order).reset_index().dropna()

            # 建立圓餅圖
            fig_bar = px.pie(df_bar_sorted, 
                            values='count', 
                            names='Zoning',
                            labels={'Zoning': '分區', 'count': '交易數量'},
                            hole=0.4,
                            color='Zoning',
                            color_discrete_map=zoning_color_map,
                            category_orders={'Zoning': zoning_order})

            # 更新圖表樣式
            fig_bar.update_traces(
                textposition='auto', 
                textinfo='label+value+percent', 
                textfont_size=14, 
                texttemplate='%{label}<br>%{value}筆<br>%{percent}', 
                rotation=90
            )

            # 添加中心標註
            fig_bar.add_annotation(
                text=f"總筆數<br>{total_count}筆", 
                x=0.5, y=0.5, 
                font_size=22, 
                showarrow=False
            )

            # 更新布局
            fig_bar.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        with tab2:
            zoning_color_map = {
                'R1': '#FF6B6B',
                'R2': '#45B7D1', 
                'R3': '#FFA07A',
                'R4': '#F7DC6F',
                'R5': '#BB8FCE'
            }

            fig_line = px.line(df_bar2, x='yyyy', y='mean', color='Zoning',color_discrete_map=zoning_color_map,
                       labels={'mean': '平均單價(萬/坪)', 'yyyy': '年度', 'Zoning': '建物類別'},
                       markers=True, text='mean')
            fig_line.update_traces(texttemplate='%{text:.1f}', textposition="top center")
            fig_line.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_line, use_container_width=True)


######################################################################################################################################################

col1, col2 = st.columns(2)
#########################
### 交易量&平均單價 樹狀 ###
#########################

with col1:
    st.subheader(f":blue[🌲{quarter[:4]}年第{quarter[5:]}季 - 交易量&平均單價 樹狀圖]", anchor =False, divider='blue')
    
    if dist == '全區':
        df3 = st.session_state.data[st.session_state.data['Quarter'] == quarter]
        df_treemap = df3.groupby(['Greater Sydney District','Council']).agg(
                count=('UNITS', 'count'),
                mean =('UNITS', 'mean')).reset_index()
        df_treemap['mean'] = df_treemap['mean'].round(2)
    else:
        df3 = st.session_state.data[(st.session_state.data['Quarter'] == quarter) & (st.session_state.data['Greater Sydney District'] == dist)]
        
        df_treemap = df3.groupby(['Council']).agg(
                count=('UNITS', 'count'),
                mean =('UNITS', 'mean')).reset_index()
        df_treemap['mean'] = df_treemap['mean'].round(2)

    with st.container(border=True):
        fig_treemap = px.treemap(df_treemap, 
            path=['Council'] if dist != '全區' else ['Greater Sydney District', 'Council'],
            values='count',
            color='mean',
            color_continuous_scale='Blues',
            labels={'count': '交易數量', 'mean': '平均單價(萬/坪)', 'Council': '地方政府'},
            hover_data={'mean': ':.2f'})
        fig_treemap.update_traces(texttemplate='<b>%{label}</b><br>交易數量: %{value}<br>平均單價: %{customdata[0]:.2f}')
        fig_treemap.update_layout(
            font_size=12,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(
            orientation="h",
            yanchor="bottom", 
            y=-0.3,
            xanchor="center", 
            x=0.5
            )
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

        st.caption("面積大小代表交易量，顏色深淺代表平均單價")


###################
### 結算期間分布  ###
###################
with col2:
    st.subheader(f":blue[📅{quarter[:4]}年第{quarter[5:]}季 - 結算期間分布]", anchor =False, divider='blue')
   
    if dist != '全區':
        df4 = st.session_state.data[(st.session_state.data['Quarter'] == quarter) & (st.session_state.data['Greater Sydney District'] == dist)]
    else:
        df4 = st.session_state.data[st.session_state.data['Quarter'] == quarter]  
    df_zoning = df4.groupby(['Council', 'TermType']).agg(
            count=('UNITS', 'count')).reset_index()
   
    with st.container(border=True):
        
        # st.write("這裡可以放其他圖表或資訊")
        # st.dataframe(df_zoning)
        # Calculate Short-term percentage for each Council to determine order
        df_order = df4.groupby('Council')['TermType'].apply(lambda x: (x == 'Short-term').sum() / len(x) * 100).reset_index()
        df_order.columns = ['Council', 'Short_term_pct']
        df_order = df_order.sort_values('Short_term_pct')
        council_order = df_order['Council'].tolist()
        
        color_map = {'Short-term': '#98D8C8', 'Medium-term': '#FFA07A', 'Long-term': '#FF6B6B'}
        fig_grouped_bar = px.histogram(df4, x='Council', color='TermType', 
              barnorm='percent',
              color_discrete_map=color_map,
              category_orders={'TermType': ['Short-term', 'Medium-term', 'Long-term'], 'Council': council_order},
              labels={'Council': 'Council', 'TermType': '交易期間類型', 'percent': '百分比'})
        fig_grouped_bar.update_traces(texttemplate='%{y:.1f}%', textposition='auto')
        fig_grouped_bar.update_layout(
            xaxis_tickangle=-45,
            yaxis=dict(range=[0, 110]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="center", x=0.5),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_grouped_bar, use_container_width=True)

        st.caption("結算天期 = 結算日期 - 契約日期(小於等於90天為短期，90天至180天為中期，大於180天為長期)")

# st.dataframe(df_summary)
# st.dataframe(df_bar)
# st.dataframe(df_treemap)
# st.dataframe(df_zoning)


######################################################################################################################################################
# API 設定
api_key = st.sidebar.text_input('請輸入您的 OpenAI API Key')
# st.sidebar.write(api_key)
if st.sidebar.button('生成AI分析結果') :

    openai.api_key = api_key
    openai.api_base = "https://api.groq.com/openai/v1"
    model = "meta-llama/llama-4-scout-17b-16e-instruct" # 模型設定
    system_prompt = """你是一個澳洲房市的數據分析師，擅長分析澳洲雪梨地區的住宅大樓產品。我給你表格後，請將這些數據進行分析，並寫出這些表格的結論。
                        1. 用繁體中文回答，並使用台灣的習慣用語，不要出現除了繁體中文之外的語言
                        2. 回答時要簡潔明瞭，字數上限為300字
                        3. 不要列點，直接寫成一段話
                        4. 你可以使用表格中的數據來支持你的分析結論
                        5. 若有空值，請不要分析該時間點的數據，直接忽略
                        6. 主要分析數據中需要被特別注意的部分。
                        
                    """ # 系統提示詞

    final_prompt = f"""
    根據下列表格回答問題：
        
    {dist}的平均單價趨勢(欄位count為交易量、欄位mean為平均單價):{df_summary.to_dict('records')}
    {quarter}這個時間點，{dist}中不同建物類別的交易量(欄位Zoning為建物類別、欄位count為交易量):{df_bar.to_dict('records')}
    {quarter}這個時間點，{dist}中不同地方政府的交易量和單價(欄位Council為地方政府、欄位count為交易量、欄位mean為平均單價){df_treemap.to_dict('records')}
    {quarter}這個時間點，{dist}中不同地方政府的有結算期間分布(欄位Council為地方政府、欄位TermType為結算期間類別、欄位count為交易量){df_zoning.to_dict('records')}


    使用者的問題是：請針對表格內的資訊進行資料分析，判斷澳洲整體房市情形。

    請根據資料內容回覆。
    """

    try:
        # 呼叫生成式AI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt},
            ]
        )

        # 取得AI回答
        answer = response.choices[0].message.content
        
        # # 顯示結果
        # print("=== 生成式AI回答 ===")
        # print(f"問題: {user_input}")
        # print(f"回答: {answer}")
        # print(f"時間: {chat_record['timestamp']}")
        
    except Exception as e:
        error_msg = f"生成回應時發生錯誤: {str(e)}"

    def stream_data(_LOREM_IPSUM):
        for word in _LOREM_IPSUM.split(" "):
            yield word + " "
            time.sleep(0.6)

    robot = st.sidebar.chat_message("assistant")
    robot.write_stream(stream_data(answer))