import streamlit as st

# --- Arka plan rengini ayarla ---
def set_bg_color(color="#808080"):  
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Fonksiyonu çağır
set_bg_color()


import streamlit as st
from pymilvus import connections, utility, Collection
from sentence_transformers import SentenceTransformer, util # type: ignore
import requests
import torch
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
project_id = os.getenv("project_id")
MILVUS_HOST=os.getenv("MILVUS_HOST")
MILVUS_PORT=os.getenv("MILVUS_PORT")
MILVUS_USERNAME=os.getenv("MILVUS_USERNAME")
MILVUS_PASSWORD=os.getenv("MILVUS_PASSWORD")


# Milvus Bağlantısı (Sayfa yüklenirken bir kez bağlan)
@st.cache_resource
def connect_milvus():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, secure=True, user=MILVUS_USERNAME, password=MILVUS_PASSWORD)

# IBM Token Alma
@st.cache_data
def get_ibm_token(API_KEY):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = f'grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={API_KEY}'

    response = requests.post('https://iam.cloud.ibm.com/identity/token', headers=headers, data=data)

    if response.status_code != 200:
        st.error("IBM Token alınamadı.")
        st.stop()

    return response.json()["access_token"]

# Milvus'ta Şirket Belirleme ve Doküman Arama
def classify_company(user_query):
    embeddings = model.encode([user_query] + company_list)
    query_embedding = embeddings[0]
    company_embeddings = embeddings[1:]

    cosine_scores = util.cos_sim(query_embedding, company_embeddings)

    best_score = cosine_scores.max().item()
    best_index = cosine_scores.argmax().item()

    if best_score > 0.7:
        return company_list[best_index]
    else:
        return None

def search_with_auto_classification(user_query):
    company_name = classify_company(user_query)
    
    if not company_name:
        return None, "Uygun bir şirket bulunamadı. Lütfen daha açık bir şirket adı belirtin."

    available_collections = utility.list_collections()
    collection_mapping = {
        "qnb": "wx_collection_qnb",
        "garanti": "wx_collection_garanti"
    }

    if company_name not in collection_mapping:
        return None, f"Collection mapping not found for {company_name}."

    target_collection = collection_mapping[company_name]

    if target_collection not in available_collections:
        return None, f"Collection '{target_collection}' not found in Milvus."

    collection = Collection(target_collection)
    query_vector = model.encode([user_query])
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=query_vector,
        anns_field="vector",
        param=search_params,
        limit=3,
        output_fields=["document_name", "text"]
    )

    matched_docs = []
    for hits in results:
        for hit in hits:
            matched_docs.append({
                "score": hit.distance,
                "title": hit.entity.get("document_name"),
                "content": hit.entity.get("text")
            })

    return matched_docs, None

def route_to_agent(user_query, ibm_token, project_id):
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"

    routing_prompt = (
        "Aşağıdaki kullanıcı sorusu hangi analiz türüne giriyor? "
        "Sadece aşağıdaki seçeneklerden birini **tek kelime olarak** cevapla:\n"
        "- büyüme\n"
        "- pazar\n"
        "- sıralama\n"
        "- stage\n"
        "- bilanço aktif\n"
        "- bilanço pasif\n"
        "- karzarar\n"
        "- genel\n\n"
        f"Soru: {user_query}"
    )


    messages = [
        {"role": "system", "content": "Sen bir sınıflandırma ajanısın. Verilen soruyu yukarıdaki kategorilere göre etiketle."},
        {"role": "user", "content": routing_prompt}
    ]

    body = {
        "messages": messages,
        "project_id": project_id,
        "model_id": "mistralai/mistral-large",
        "frequency_penalty": 0,
        "max_tokens": 100,
        "presence_penalty": 0,
        "temperature": 0,
        "top_p": 1
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ibm_token}" 
    }

    response = requests.post(url, headers=headers, json=body)

    if response.status_code != 200:
        st.error("Routing için LLM yanıtı alınamadı.")
        st.stop()

    content = response.json()["choices"][0]["message"]["content"].strip().lower()
    return content


def generate_llm_response_with_routing(user_query, context, project_id, ibm_token):
    # 🔹 URL tanımlanıyor
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
    # 🔹 Analiz tipini belirle (routing)
    agent_type = route_to_agent(user_query, ibm_token, project_id)

    # 🔹 Gelişmiş prompt seti
    prompt_map = {
        "büyüme": (
            "Sen bir finansal büyüme analistisin. Net kâr üzerinden yıllık (YoY) ve çeyreklik (QoQ) büyüme oranlarını "
            "hem oran (%) hem de tutar (milyon TL) olarak hesapla ve göster. Tablo formatında sun. Soru içinde dönem bilgisi verilmemişse, en güncel (son açıklanan) dönemin "
            "verilerini kullan."
        ),
        "pazar": (
            "Sen bankacılık pazar payı uzmanısın. Kredilerde (tüketici/ticari, TL/YP), mevduatta (vadeli/vadesiz, TL/YP) "
            "pazar paylarını ayrıntılı şekilde analiz et. Tablo biçiminde ve sade bir sunum yap.Soru içinde dönem bilgisi verilmemişse, en güncel (son açıklanan) dönemin "
            "verilerini kullan."
        ),
        "sıralama": (
            "Sen finansal sıralama analistisin. Bankaları aktif büyüklük, net kâr/zarar, toplam kredi ve toplam mevduat "
            "gibi kriterlere göre sırala. En güncel verilere göre tabloyla sun."
        ),
        "stage": (
            "Sen canlı kredi izleme uzmanısısın.Sadece 'Birinci ve ikinci aşama krediler' veya ' Standart Nitelikli ve Yakın İzlemedeki krediler ile yeniden yapılandırılan Yakın İzlemedeki kredilere ilişkin bilgiler' başlığı altında olan tüm tabloların toplam değerlerini TL ve YP olarak ayrı ayrı getir. Soru içinde dönem bilgisi verilmemişse, en güncel (son açıklanan) dönemin verilerini kullan."
        ),
        "genel": (
            "Sen bankacılık verilerini analiz eden bir asistanısın. Kullanıcı sorusuna göre sadece gerekli cevabı ver.Soru içinde dönem bilgisi verilmemişse, en güncel (son açıklanan) dönemin "
            "verilerini kullan."
        ),
        "bilanço aktif":("Sen bilanço aktif tablosu özelinde çalışan bir asistanısın. Sorulan soru eğer 'Bilanço Aktif' veya 'Bilanço Varlıklar' ile ilgiliyse Cari Dönem ve Önceki Dönem için ayrı ayrı TP, YP ve TOPLAM şeklinde ifade edilmiş durumdadır. Bu bilgileri TABLO içerisinde eksiksiz getireceksin.Soru içinde dönem bilgisi verilmemişse, en güncel (son açıklanan) dönemin "
            "verilerini kullan."),
        "bilanço pasif":("Sen bilanço pasif tablosu özelinde çalışan bir asistanısın. Sorulan soru eğer 'Bilanço Pasif' veya 'Bilanço Yükümlülükler' ile ilgiliyse Cari Dönem ve Önceki Dönem için ayrı ayrı TP, YP ve TOPLAM şeklinde ifade edilmiş durumdadır. Bu bilgileri TABLO içerisinde eksiksiz getireceksin.Soru içinde dönem bilgisi verilmemişse, en güncel (son açıklanan) dönemin "
            "verilerini kullan."),
        "karzarar":("Sen Kar veya Zarar Tablosu tablosu özelinde çalışan bir asistanısın. Sorulan soru eğer 'Kar veya Zarar Tablosu' ile ilgiliyse Cari Dönem ve Önceki Dönem için ayrı ayrı TP, YP ve TOPLAM şeklinde ifade edilmiş durumdadır. Bu bilgileri TABLO içerisinde eksiksiz getireceksin.Soru içinde dönem bilgisi verilmemişse, en güncel (son açıklanan) dönemin "
            "verilerini kullan.")
    }

    # 🔹 Prompta karar ver
    system_prompt = prompt_map.get(agent_type, prompt_map["genel"])

    # 🔹 Dokümanları metne çevir
    context_text = "\n\n".join([
        f"Belge: {doc['title']}\nİçerik: {doc['content']}" for doc in context
    ])

    # 🔹 LLM mesajları
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Kullanıcı sorusu: {user_query}\n\nİlgili belgeler:\n{context_text}"}
    ]

    body = {
        "messages": messages,
        "project_id": project_id,
        "model_id": "mistralai/mistral-large",
        "frequency_penalty": 0,
        "max_tokens": 6000,
        "presence_penalty": 0,
        "temperature": 0,
        "top_p": 1
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ibm_token}"
    }

    # 🔹 POST isteği
    response = requests.post(url, headers=headers, json=body)

    if response.status_code != 200:
        st.error("LLM yanıt alınamadı.")
        st.stop()

    return response.json()


# Uygulama Başlangıcı
st.set_page_config(page_title="Milvus + LLM Chat", page_icon="🤖")

st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/95/TEB_LOGO.png" alt="Logo" width="120" style="margin-right:10px;">
    </div>
    """,
    unsafe_allow_html=True
)
st.title("CepTEBot")

connect_milvus()


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# Model ve Şirketler
model = SentenceTransformer(
    "intfloat/multilingual-e5-large", device=device)
company_list = ["qnb", "garanti"]

# IBM API Key ve Project ID
#api_key = st.secrets["IBM_API_KEY"]  # streamlit secrets kullanılabilir
#project_id = st.secrets["PROJECT_ID"]

user_query = st.text_input("Lütfen sorunuz:")

if st.button("Sorgula") and user_query.strip() != "":
    with st.spinner("Milvus'tan doküman aranıyor..."):
        docs, error = search_with_auto_classification(user_query)

    if error:
        st.error(error)
    elif not docs:
        st.warning("Eşleşen doküman bulunamadı.")
    else:
        st.success("İlgili dokümanlar bulundu!")
        with st.spinner("LLM routing yapılıyor..."):
            ibm_token = get_ibm_token(API_KEY)
            llm_response = generate_llm_response_with_routing(user_query, docs, project_id, ibm_token)
        
        st.subheader("💬 LLM Yanıtı")
        st.write(llm_response["choices"][0]["message"]["content"])
        


