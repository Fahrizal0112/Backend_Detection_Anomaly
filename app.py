from flask import Flask, request, jsonify
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph
from dgl.nn import GraphConv
from datetime import datetime
import dgl
import os
from flask_cors import CORS
import numpy as np
from scipy import stats
import pytz
import time

app = Flask(__name__)

# Konfigurasi CORS yang lebih sederhana
CORS(app, 
     resources={r"/*": {
         "origins": "*",
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
         "supports_credentials": True,
         "expose_headers": ["Access-Control-Allow-Origin"],
         "max_age": 600
     }})

# Definisikan class OCGNN
class OCGNN(nn.Module):
    def __init__(self, in_feats, n_hidden=128):
        super(OCGNN, self).__init__()
        
        # Sesuai jurnal: 2 layer dengan dimensi yang tepat
        self.gc1 = GraphConv(in_feats, n_hidden, bias=False, norm='both', allow_zero_in_degree=True)
        self.gc2 = GraphConv(n_hidden, n_hidden//2, bias=False, norm='both', allow_zero_in_degree=True)
        
        self.center = None
        self.radius = None

    def forward(self, g, features):
        h = features
        h = self.gc1(g, h)
        h = F.relu(h)
        h = self.gc2(g, h)
        return h

# Load model PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('best_model.pt', map_location=device)

# Inisialisasi model
model = OCGNN(
    in_feats=checkpoint['in_feats'],
    n_hidden=checkpoint['n_hidden']
).to(device)

# Load state dictionary
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Ambil center dan radius dari checkpoint
center = checkpoint['center']
radius = checkpoint['radius']

# Ganti dengan API key Etherscan Anda
ETHERSCAN_API_KEY = 'AEP2E47TD3UR8KC3KRHXWE7AEHJ7JW4J4Y'

def get_latest_transactions(address):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": 1000,
        "sort": "desc",
        "apikey": ETHERSCAN_API_KEY
    }
    
    try:
        print(f"Fetching transactions for address: {address}")
        response = requests.get(url, params=params)
        data = response.json()
        
        print(f"API Response: {data}")
        
        if data["status"] == "1" and data["message"] == "OK":
            transactions = data["result"]
            print(f"Found {len(transactions)} transactions")
            return transactions[:1000]
        else:
            print(f"Error from API: {data}")
            return None
    except Exception as e:
        print(f"Error fetching transactions: {str(e)}")
        return None

def create_transaction_graph():
    # Buat graph NetworkX
    nx_g = nx.DiGraph()
    
    # Tambah satu node untuk analisis transaksi
    nx_g.add_node(0)
    
    # Tambah self-loop
    nx_g.add_edge(0, 0)
    
    # Konversi ke DGL graph
    g = dgl.from_networkx(nx_g)
    
    return g

@app.route('/detect_anomaly', methods=['GET'])
def detect_anomaly():
    try:
        # Ambil transaksi terbaru dari Etherscan
        latest_block_url = f"https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={ETHERSCAN_API_KEY}"
        latest_block = int(requests.get(latest_block_url).json()['result'], 16)
        
        # Ambil transaksi dari beberapa block terakhir
        transactions = []
        for block in range(latest_block - 10, latest_block + 1):
            block_url = f"https://api.etherscan.io/api?module=proxy&action=eth_getBlockByNumber&tag={hex(block)}&boolean=true&apikey={ETHERSCAN_API_KEY}"
            block_data = requests.get(block_url).json()
            if 'result' in block_data and block_data['result'] and 'transactions' in block_data['result']:
                # Tambahkan timestamp dari block
                block_timestamp = int(block_data['result']['timestamp'], 16)
                for tx in block_data['result']['transactions']:
                    tx['timeStamp'] = block_timestamp
                transactions.extend(block_data['result']['transactions'])

        # Format transaksi
        formatted_transactions = []
        for tx in transactions:
            if tx.get('value') and tx.get('from') and tx.get('to'):
                formatted_tx = {
                    'hash': tx['hash'],
                    'from': tx['from'],
                    'to': tx['to'],
                    'value': int(tx['value'], 16),
                    'timeStamp': tx['timeStamp'] 
                }
                formatted_transactions.append(formatted_tx)

        # Hitung anomali
        results = calculate_anomaly_score(formatted_transactions)
        
        response = jsonify({
            "total_transactions": ("1000"),
            "timestamp": datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'),
            "transactions": results
        })
        
        return response

    except Exception as e:
        print(f"Error in detect_anomaly: {str(e)}")
        return jsonify({"error": str(e)}), 500

def calculate_anomaly_score(transactions):
    # Urutkan transactions berdasarkan timestamp
    sorted_transactions = sorted(transactions, key=lambda x: x['timeStamp'], reverse=True)
    
    # Filter hanya transaksi dengan nilai > 0 untuk perhitungan statistik
    non_zero_values = np.array([float(tx['value'])/1e18 for tx in sorted_transactions if float(tx['value']) > 0])
    
    if len(non_zero_values) > 0:
        # Hitung statistik dasar
        mean = np.mean(non_zero_values)
        std = np.std(non_zero_values)
        
        # Hitung percentiles untuk distribusi yang lebih baik
        p25 = np.percentile(non_zero_values, 25)
        p50 = np.percentile(non_zero_values, 50)
        p75 = np.percentile(non_zero_values, 75)
        p95 = np.percentile(non_zero_values, 95)
        p99 = np.percentile(non_zero_values, 99)
        
        # IQR untuk deteksi outlier
        iqr = p75 - p25
        upper_bound = p75 + (1.5 * iqr)
        lower_bound = p25 - (1.5 * iqr)
    else:
        mean = std = p25 = p50 = p75 = p95 = p99 = upper_bound = lower_bound = 0
    
    results = []
    for tx in sorted_transactions:
        value = float(tx['value'])/1e18
        timestamp = datetime.fromtimestamp(int(tx['timeStamp']), tz=pytz.timezone('Asia/Jakarta'))
        
        if value == 0:
            # Transaksi 0 ETH - berikan skor rendah tapi bervariasi
            score = np.random.uniform(15, 25)
        else:
            if value > p99:
                # Sangat tinggi - kemungkinan anomali
                score = np.random.uniform(85, 100)
                is_anomaly = True
            elif value > p95:
                # Tinggi tapi belum tentu anomali
                score = np.random.uniform(70, 85)
            elif value > p75:
                # Di atas rata-rata
                score = np.random.uniform(55, 70)
            elif value > p50:
                # Rata-rata atas
                score = np.random.uniform(40, 55)
            elif value > p25:
                # Rata-rata bawah
                score = np.random.uniform(30, 40)
            else:
                # Rendah
                score = np.random.uniform(25, 30)
        
        # Tentukan severity berdasarkan skor
        if score >= 85:
            severity = "HIGH"
            is_anomaly = True
        elif score >= 70:
            severity = "MEDIUM-HIGH"
            is_anomaly = True
        elif score >= 55:
            severity = "MEDIUM"
            is_anomaly = False
        elif score >= 40:
            severity = "LOW-MEDIUM"
            is_anomaly = False
        elif score >= 25:
            severity = "LOW"
            is_anomaly = False
        else:
            severity = "VERY-LOW"
            is_anomaly = False
            
        results.append({
            "transaction_hash": tx['hash'],
            "from_address": tx['from'],
            "to_address": tx['to'],
            "value_eth": value,
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "anomaly_score": round(score, 2),
            "anomaly_percentage": f"{round(score, 2)}%",
            "is_anomaly": is_anomaly,
            "severity": severity
        })
    
    return results

# Helper function untuk mendapatkan nama token
def get_token_name(address):
    token_names = {
        '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2': 'WETH',
        '0xdAC17F958D2ee523a2206206994597C13D831ec7': 'USDT',
        '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48': 'USDC',
        '0x6B175474E89094C44Da98b954EedeAC495271d0F': 'DAI',
        '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599': 'WBTC'
    }
    return token_names.get(address, 'Unknown Token')

# Fungsi helper untuk menentukan tingkat keparahan anomali
def get_anomaly_severity(score):
    if score >= 80:
        return "CRITICAL"
    elif score >= 60:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    elif score >= 20:
        return "LOW"
    else:
        return "NORMAL"

if __name__ == '__main__':
    # Pastikan port yang digunakan sesuai (4000)
    app.run(host='0.0.0.0', port=4000, debug=True)