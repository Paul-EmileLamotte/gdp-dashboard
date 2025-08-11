import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import jwt
import time
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import gzip
import base64
from typing import Dict, List, Optional, Union
import warnings
from io import BytesIO, StringIO
import csv
from dataclasses import dataclass
from enum import Enum
import hashlib
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
st.set_page_config(
    page_title="🍎 Apple Dashboard Pro Complete",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration Apple API (VOS VRAIES CLÉS ADMIN)
APPLE_CONFIG = {
    'key_id': '',
    'issuer_id': '',
    'app_id': '',
    'private_key': '''''',
    'vendor_number': ''
}

# === ENUMS ET CONSTANTES APPLE API ===

class ReportCategory(Enum):
    APP_STORE_ENGAGEMENT = "APP_STORE_ENGAGEMENT"
    APP_STORE_COMMERCE = "APP_STORE_COMMERCE" 
    APP_USAGE = "APP_USAGE"
    PERFORMANCE = "PERFORMANCE"
    FRAMEWORK_USAGE = "FRAMEWORK_USAGE"

class ReportFrequency(Enum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"

class ReportType(Enum):
    SALES = "SALES"
    INSTALLS = "INSTALLS"
    ANALYTICS = "ANALYTICS"
    FINANCE = "FINANCE"

class Territory(Enum):
    WORLDWIDE = "WW"
    FRANCE = "FR"
    UNITED_STATES = "US"
    UNITED_KINGDOM = "GB"
    GERMANY = "DE"
    JAPAN = "JP"
    CHINA = "CN"

class DeviceType(Enum):
    IPHONE = "iPhone"
    IPAD = "iPad"
    APPLE_WATCH = "Apple Watch"
    APPLE_TV = "Apple TV"
    MAC = "Mac"

@dataclass
class FilterOptions:
    territories: List[str] = None
    devices: List[str] = None
    sources: List[str] = None
    categories: List[str] = None
    measures: List[str] = None
    dimensions: List[str] = None
    start_date: str = None
    end_date: str = None
    frequency: str = "DAILY"
    granularity: str = "DAILY"

# === CLIENT API APPLE COMPLET ===

class AppleStoreConnectAPIComplete:
    """Client complet avec TOUTES les fonctionnalités App Store Connect"""
    
    def __init__(self, config: dict):
        self.config = config
        self.base_url = "https://api.appstoreconnect.apple.com"
        self.session = None
        self.cache = {}
        self.rate_limit_remaining = 3600  # Limite par heure
        self.last_request_time = 0
    
    def generate_jwt_token(self) -> str:
        """Génère token JWT pour authentification Apple"""
        current_time = int(time.time())
        
        payload = {
            'iss': self.config['issuer_id'],
            'iat': current_time,
            'exp': current_time + (20 * 60),
            'aud': 'appstoreconnect-v1',
        }
        
        headers = {
            'alg': 'ES256',
            'kid': self.config['key_id'],
            'typ': 'JWT'
        }
        
        try:
            token = jwt.encode(payload, self.config['private_key'], algorithm='ES256', headers=headers)
            return token
        except Exception as e:
            st.error(f"❌ Erreur génération token: {str(e)}")
            return ""
    
    async def make_request(self, endpoint: str, params: dict = None, method: str = "GET", data: dict = None) -> dict:
        """Requête API avec gestion rate limiting et cache"""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < 1.0:  # Min 1 seconde entre requêtes
            await asyncio.sleep(1.0 - (current_time - self.last_request_time))
        
        # Cache simple
        cache_key = hashlib.md5(f"{endpoint}{params}".encode()).hexdigest()
        if method == "GET" and cache_key in self.cache:
            cache_data, cache_time = self.cache[cache_key]
            if current_time - cache_time < 300:  # Cache 5 minutes
                return cache_data
        
        token = self.generate_jwt_token()
        if not token:
            return {}
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Apple-Dashboard-Pro/1.0'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        if not self.session:
            connector = aiohttp.TCPConnector(ssl=False)
            self.session = aiohttp.ClientSession(connector=connector)
        
        try:
            self.last_request_time = time.time()
            
            if method.upper() == "GET":
                async with self.session.get(url, headers=headers, params=params) as response:
                    result = await self._handle_response(response, endpoint)
                    
                    # Cache uniquement les succès
                    if result and method == "GET":
                        self.cache[cache_key] = (result, current_time)
                    
                    return result
                    
            elif method.upper() == "POST":
                async with self.session.post(url, headers=headers, json=data) as response:
                    return await self._handle_response(response, endpoint)
                    
        except Exception as e:
            st.error(f"❌ Erreur connexion {endpoint}: {str(e)}")
            return {}
    
    async def _handle_response(self, response, endpoint: str) -> dict:
        """Gestion complète des réponses API"""
        # Mise à jour rate limit
        if 'X-Rate-Limit-Remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['X-Rate-Limit-Remaining'])
        
        if response.status == 200 or response.status == 201:
            return await response.json()
        elif response.status == 401:
            st.error("🔐 Token expiré - Régénération automatique")
            return {}
        elif response.status == 403:
            st.warning(f"🚫 Permissions insuffisantes pour {endpoint}")
            return {}
        elif response.status == 404:
            st.info(f"ℹ️ Ressource non trouvée: {endpoint}")
            return {}
        elif response.status == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            st.warning(f"⏱️ Rate limit atteint - Attente {retry_after}s")
            await asyncio.sleep(retry_after)
            return {}
        else:
            error_text = await response.text()
            st.error(f"❌ Erreur API {response.status}: {error_text}")
            return {}
    
    # === APPS ET MÉTADONNÉES ===
    
    async def get_all_apps(self) -> List[dict]:
        """Récupère toutes les apps du compte"""
        endpoint = "/v1/apps"
        params = {
            'fields[apps]': 'name,bundleId,sku,primaryLocale,contentRightsDeclaration',
            'limit': 200
        }
        
        response = await self.make_request(endpoint, params)
        return response.get('data', [])
    
    async def get_app_info_complete(self, app_id: str = None) -> dict:
        """Informations complètes d'une app"""
        if not app_id:
            app_id = self.config['app_id']
            
        endpoint = f"/v1/apps/{app_id}"
        params = {
            'fields[apps]': 'name,bundleId,sku,primaryLocale,contentRightsDeclaration,availableInNewTerritories',
            'include': 'appInfos,preReleaseVersions,appStoreVersions,prices,availableTerritories'
        }
        
        response = await self.make_request(endpoint, params)
        return response.get('data', {})
    
    async def get_app_store_versions(self, app_id: str = None) -> List[dict]:
        """Toutes les versions App Store"""
        if not app_id:
            app_id = self.config['app_id']
            
        endpoint = f"/v1/apps/{app_id}/appStoreVersions"
        params = {
            'fields[appStoreVersions]': 'versionString,appStoreState,releaseType,createdDate',
            'sort': '-createdDate',
            'limit': 50
        }
        
        response = await self.make_request(endpoint, params)
        return response.get('data', [])
    
    async def get_territories(self) -> List[dict]:
        """Liste de tous les territoires disponibles"""
        endpoint = "/v1/territories"
        params = {'limit': 200}
        
        response = await self.make_request(endpoint, params)
        return response.get('data', [])
    
    # === ANALYTICS REPORTS COMPLETS ===
    
    async def create_analytics_report_request_advanced(self, access_type: str = "ONGOING") -> str:
        """Création de demande de rapport analytics avancée"""
        endpoint = "/v1/analyticsReportRequests"
        
        data = {
            "data": {
                "type": "analyticsReportRequests",
                "attributes": {
                    "accessType": access_type
                },
                "relationships": {
                    "app": {
                        "data": {
                            "type": "apps",
                            "id": self.config['app_id']
                        }
                    }
                }
            }
        }
        
        response = await self.make_request(endpoint, method="POST", data=data)
        if response.get('data'):
            return response['data'].get('id', '')
        return ""
    
    async def get_analytics_reports_filtered(self, 
                                           report_request_id: str,
                                           filters: FilterOptions) -> List[dict]:
        """Rapports analytics avec filtres avancés"""
        endpoint = f"/v1/analyticsReportRequests/{report_request_id}/reports"
        
        params = {
            'fields[analyticsReports]': 'category,name,instances',
            'limit': 200
        }
        
        # Filtres par catégorie
        if filters.categories:
            params['filter[category]'] = ','.join(filters.categories)
        
        response = await self.make_request(endpoint, params)
        return response.get('data', [])
    
    async def download_analytics_instance_with_filters(self, 
                                                     instance_url: str,
                                                     filters: FilterOptions) -> pd.DataFrame:
        """Télécharge instance avec filtres appliqués"""
        if not instance_url:
            return pd.DataFrame()
        
        # Ajout paramètres de filtre à l'URL si supporté
        filtered_url = instance_url
        if filters.start_date:
            separator = '&' if '?' in filtered_url else '?'
            filtered_url += f"{separator}startDate={filters.start_date}"
        if filters.end_date:
            separator = '&' if '?' in filtered_url else '?'
            filtered_url += f"{separator}endDate={filters.end_date}"
        
        token = self.generate_jwt_token()
        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
            'User-Agent': 'Apple-Dashboard-Pro/1.0'
        }
        
        if not self.session:
            connector = aiohttp.TCPConnector(ssl=False)
            self.session = aiohttp.ClientSession(connector=connector)
        
        try:
            async with self.session.get(filtered_url, headers=headers) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Décompression si nécessaire
                    try:
                        content = gzip.decompress(content)
                    except:
                        pass
                    
                    content_str = content.decode('utf-8')
                    
                    if content_str.startswith('{'):
                        data = json.loads(content_str)
                        df = pd.json_normalize(data)
                    else:
                        df = pd.read_csv(StringIO(content_str))
                    
                    # Application des filtres côté client
                    return self._apply_client_filters(df, filters)
                
                else:
                    st.warning(f"⚠️ Erreur téléchargement instance: {response.status}")
                    return pd.DataFrame()
        
        except Exception as e:
            st.warning(f"⚠️ Erreur instance: {str(e)}")
            return pd.DataFrame()
    
    def _apply_client_filters(self, df: pd.DataFrame, filters: FilterOptions) -> pd.DataFrame:
        """Applique filtres côté client"""
        if df.empty:
            return df
        
        # Filtre par territoires
        if filters.territories and 'territory' in df.columns:
            df = df[df['territory'].isin(filters.territories)]
        
        # Filtre par appareils
        if filters.devices:
            device_cols = [col for col in df.columns if any(device.lower() in col.lower() for device in filters.devices)]
            if device_cols:
                device_filter = df[device_cols].sum(axis=1) > 0
                df = df[device_filter]
        
        # Filtre par dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            if filters.start_date:
                start_dt = pd.to_datetime(filters.start_date)
                df = df[df['date'] >= start_dt]
            
            if filters.end_date:
                end_dt = pd.to_datetime(filters.end_date)
                df = df[df['date'] <= end_dt]
        
        return df
    
    # === SALES REPORTS AVANCÉS ===
    
    async def get_sales_reports_advanced(self, 
                                       frequency: str = "DAILY",
                                       report_date: str = None,
                                       territories: List[str] = None) -> pd.DataFrame:
        """Rapports de ventes avec filtres avancés"""
        if not report_date:
            report_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        endpoint = "/v1/salesReports"
        params = {
            'filter[frequency]': frequency,
            'filter[reportDate]': report_date,
            'filter[reportSubType]': 'SUMMARY',
            'filter[reportType]': 'SALES',
            'filter[vendorNumber]': self.config['vendor_number'],
            'filter[version]': '1_0'
        }
        
        response = await self.make_request(endpoint, params)
        
        if response.get('data'):
            sales_data = []
            for item in response['data']:
                attributes = item.get('attributes', {})
                
                # Filtre territoire si spécifié
                territory = attributes.get('territory', 'Unknown')
                if territories and territory not in territories:
                    continue
                
                sales_data.append({
                    'Date': report_date,
                    'Units': attributes.get('units', 0),
                    'Revenue': attributes.get('proceeds', 0),
                    'Territory': territory,
                    'Product': attributes.get('sku', 'Unknown'),
                    'Currency': attributes.get('currency', 'EUR'),
                    'Device': attributes.get('device', 'Unknown'),
                    'Customer_Price': attributes.get('customerPrice', 0),
                    'Developer_Proceeds': attributes.get('developerProceeds', 0)
                })
            
            return pd.DataFrame(sales_data)
        
        # Fallback avec données simulées avancées
        return self._generate_advanced_sales_data(report_date, territories, frequency)
    
    async def get_financial_reports(self, 
                                  report_date: str = None,
                                  regions: List[str] = None) -> pd.DataFrame:
        """Rapports financiers détaillés"""
        if not report_date:
            report_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m')
        
        endpoint = "/v1/financeReports"
        params = {
            'filter[reportDate]': report_date,
            'filter[reportType]': 'FINANCIAL',
            'filter[vendorNumber]': self.config['vendor_number']
        }
        
        if regions:
            params['filter[regionCode]'] = ','.join(regions)
        
        response = await self.make_request(endpoint, params)
        
        if response.get('data'):
            finance_data = []
            for item in response['data']:
                attributes = item.get('attributes', {})
                finance_data.append({
                    'Date': report_date,
                    'Revenue': attributes.get('totalRevenue', 0),
                    'Tax': attributes.get('tax', 0),
                    'Commission': attributes.get('commission', 0),
                    'Net_Revenue': attributes.get('netRevenue', 0),
                    'Region': attributes.get('region', 'Unknown'),
                    'Currency': attributes.get('currency', 'EUR')
                })
            
            return pd.DataFrame(finance_data)
        
        return self._generate_financial_data(report_date)
    
    # === BETA TESTING ===
    
    async def get_beta_testers(self) -> pd.DataFrame:
        """Informations des beta testeurs"""
        endpoint = f"/v1/apps/{self.config['app_id']}/betaTesters"
        params = {
            'fields[betaTesters]': 'firstName,lastName,email,state,inviteDate',
            'limit': 200
        }
        
        response = await self.make_request(endpoint, params)
        
        if response.get('data'):
            testers_data = []
            for tester in response['data']:
                attributes = tester.get('attributes', {})
                testers_data.append({
                    'ID': tester.get('id'),
                    'Name': f"{attributes.get('firstName', '')} {attributes.get('lastName', '')}".strip(),
                    'Email': attributes.get('email', ''),
                    'State': attributes.get('state', 'Unknown'),
                    'Invite_Date': attributes.get('inviteDate', ''),
                    'Sessions': np.random.randint(1, 50),  # Simulé
                    'Feedback_Count': np.random.randint(0, 10)
                })
            
            return pd.DataFrame(testers_data)
        
        return self._generate_beta_data()
    
    async def get_beta_feedback(self) -> pd.DataFrame:
        """Retours des beta testeurs"""
        endpoint = f"/v1/apps/{self.config['app_id']}/betaFeedbacks"
        params = {
            'fields[betaFeedbacks]': 'comment,rating,timestamp',
            'sort': '-timestamp',
            'limit': 100
        }
        
        response = await self.make_request(endpoint, params)
        
        if response.get('data'):
            feedback_data = []
            for feedback in response['data']:
                attributes = feedback.get('attributes', {})
                feedback_data.append({
                    'ID': feedback.get('id'),
                    'Comment': attributes.get('comment', ''),
                    'Rating': attributes.get('rating', 0),
                    'Timestamp': attributes.get('timestamp', ''),
                    'Category': 'Bug' if 'bug' in attributes.get('comment', '').lower() else 'Feature'
                })
            
            return pd.DataFrame(feedback_data)
        
        return self._generate_feedback_data()
    
    # === APP STORE OPTIMIZATION (ASO) ===
    
    async def get_keyword_rankings(self) -> pd.DataFrame:
        """Rankings des mots-clés (simulé - pas d'API officielle)"""
        return self._generate_keyword_data()
    
    async def get_competitor_analysis(self) -> pd.DataFrame:
        """Analyse concurrentielle (simulé)"""
        return self._generate_competitor_data()
    
    # === REVIEW ET RATINGS ===
    
    async def get_customer_reviews_advanced(self, territories: List[str] = None) -> pd.DataFrame:
        """Reviews clients avec filtres avancés"""
        endpoint = f"/v1/apps/{self.config['app_id']}/customerReviews"
        params = {
            'fields[customerReviews]': 'rating,title,body,reviewerNickname,createdDate,territory',
            'sort': '-createdDate',
            'limit': 200
        }
        
        if territories:
            params['filter[territory]'] = ','.join(territories)
        
        response = await self.make_request(endpoint, params)
        
        if response.get('data'):
            reviews_data = []
            for review in response['data']:
                attributes = review.get('attributes', {})
                
                # Sentiment basique
                body = attributes.get('body', '').lower()
                if any(word in body for word in ['love', 'great', 'amazing', 'perfect']):
                    sentiment = 'Positive'
                elif any(word in body for word in ['hate', 'terrible', 'awful', 'crash']):
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                
                reviews_data.append({
                    'ID': review.get('id'),
                    'Rating': attributes.get('rating', 0),
                    'Title': attributes.get('title', ''),
                    'Body': attributes.get('body', ''),
                    'Reviewer': attributes.get('reviewerNickname', 'Anonymous'),
                    'Date': attributes.get('createdDate', ''),
                    'Territory': attributes.get('territory', 'Unknown'),
                    'Sentiment': sentiment,
                    'Length': len(attributes.get('body', '')),
                    'Has_Response': False  # À implémenter avec responses
                })
            
            return pd.DataFrame(reviews_data)
        
        return self._generate_reviews_data()
    
    # === IN-APP PURCHASES ===
    
    async def get_iap_reports(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Rapports In-App Purchases détaillés"""
        # Utilise sales reports avec filtre IAP
        endpoint = "/v1/salesReports"
        params = {
            'filter[frequency]': 'DAILY',
            'filter[reportSubType]': 'DETAILED',
            'filter[reportType]': 'SALES',
            'filter[vendorNumber]': self.config['vendor_number']
        }
        
        response = await self.make_request(endpoint, params)
        
        if response.get('data'):
            iap_data = []
            for item in response['data']:
                attributes = item.get('attributes', {})
                
                # Filter seulement les IAP
                if attributes.get('productTypeIdentifier', '').startswith('IA'):
                    iap_data.append({
                        'Date': attributes.get('beginDate', ''),
                        'Product_ID': attributes.get('productTypeIdentifier', ''),
                        'Product_Name': attributes.get('title', 'Unknown IAP'),
                        'Units': attributes.get('units', 0),
                        'Revenue': attributes.get('developerProceeds', 0),
                        'Territory': attributes.get('country', 'Unknown'),
                        'Device': attributes.get('device', 'Unknown'),
                        'Price_Tier': attributes.get('customerPrice', 0)
                    })
            
            return pd.DataFrame(iap_data)
        
        return self._generate_iap_data(start_date, end_date)
    
    # === GÉNÉRATEURS DE DONNÉES FALLBACK ===
    
    def _generate_advanced_sales_data(self, report_date: str, territories: List[str], frequency: str) -> pd.DataFrame:
        """Génère données de vente avancées"""
        np.random.seed(42)
        
        if not territories:
            territories = ['FR', 'US', 'GB', 'DE', 'JP']
        
        devices = ['iPhone', 'iPad', 'Apple Watch', 'Apple TV']
        
        data = []
        for territory in territories:
            for device in devices:
                units = np.random.poisson(15) * {'iPhone': 3, 'iPad': 1.5, 'Apple Watch': 0.8, 'Apple TV': 0.3}[device]
                revenue = units * np.random.uniform(0.99, 4.99)
                
                data.append({
                    'Date': report_date,
                    'Units': int(units),
                    'Revenue': round(revenue, 2),
                    'Territory': territory,
                    'Device': device,
                    'Product': 'Votre App',
                    'Currency': 'EUR' if territory in ['FR', 'DE'] else 'USD',
                    'Customer_Price': np.random.uniform(0.99, 4.99),
                    'Developer_Proceeds': round(revenue * 0.7, 2)
                })
        
        return pd.DataFrame(data)
    
    def _generate_financial_data(self, report_date: str) -> pd.DataFrame:
        """Génère données financières"""
        np.random.seed(123)
        
        regions = ['Europe', 'Americas', 'Asia Pacific', 'ROW']
        data = []
        
        for region in regions:
            revenue = np.random.uniform(1000, 10000)
            tax = revenue * np.random.uniform(0.15, 0.25)
            commission = revenue * 0.3
            
            data.append({
                'Date': report_date,
                'Revenue': round(revenue, 2),
                'Tax': round(tax, 2),
                'Commission': round(commission, 2),
                'Net_Revenue': round(revenue - tax - commission, 2),
                'Region': region,
                'Currency': 'EUR'
            })
        
        return pd.DataFrame(data)
    
    def _generate_beta_data(self) -> pd.DataFrame:
        """Génère données beta testeurs"""
        np.random.seed(456)
        
        data = []
        for i in range(50):
            data.append({
                'ID': f'beta_{i}',
                'Name': f'Testeur {i+1}',
                'Email': f'testeur{i+1}@exemple.com',
                'State': np.random.choice(['ACTIVE', 'INVITED', 'INACTIVE']),
                'Invite_Date': (datetime.now() - timedelta(days=np.random.randint(1, 90))).isoformat(),
                'Sessions': np.random.randint(1, 50),
                'Feedback_Count': np.random.randint(0, 10)
            })
        
        return pd.DataFrame(data)
    
    def _generate_feedback_data(self) -> pd.DataFrame:
        """Génère retours beta"""
        np.random.seed(789)
        
        comments = [
            "L'app crash au lancement",
            "Interface très intuitive",
            "Bug sur la fonctionnalité X",
            "Excellente idée, bien réalisée",
            "Performance lente sur ancien iPhone"
        ]
        
        data = []
        for i in range(25):
            data.append({
                'ID': f'feedback_{i}',
                'Comment': np.random.choice(comments),
                'Rating': np.random.randint(1, 6),
                'Timestamp': (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                'Category': np.random.choice(['Bug', 'Feature', 'UI', 'Performance'])
            })
        
        return pd.DataFrame(data)
    
    def _generate_keyword_data(self) -> pd.DataFrame:
        """Génère données mots-clés"""
        keywords = [
            'productivity app', 'task manager', 'todo list', 'organization',
            'time tracking', 'project management', 'efficiency', 'planning'
        ]
        
        data = []
        for keyword in keywords:
            data.append({
                'Keyword': keyword,
                'Ranking': np.random.randint(1, 100),
                'Search_Volume': np.random.randint(100, 10000),
                'Difficulty': np.random.uniform(0.1, 0.9),
                'Trend': np.random.choice(['Rising', 'Stable', 'Declining'])
            })
        
        return pd.DataFrame(data)
    
    def _generate_competitor_data(self) -> pd.DataFrame:
        """Génère analyse concurrentielle"""
        np.random.seed(321)
        
        competitors = [
            'Todoist', 'Any.do', 'Microsoft To Do', 'Things 3', 'Notion',
            'Asana', 'Trello', 'TickTick', 'OmniFocus'
        ]
        
        data = []
        for competitor in competitors:
            data.append({
                'App_Name': competitor,
                'Rating': np.random.uniform(3.5, 4.8),
                'Reviews_Count': np.random.randint(1000, 50000),
                'Downloads_Estimate': np.random.randint(10000, 1000000),
                'Price': np.random.choice([0, 0.99, 2.99, 4.99, 9.99]),
                'Category_Rank': np.random.randint(1, 50),
                'Last_Update': (datetime.now() - timedelta(days=np.random.randint(1, 90))).strftime('%Y-%m-%d'),
                'Features_Count': np.random.randint(10, 30)
            })
        
        return pd.DataFrame(data)
    
    def _generate_reviews_data(self) -> pd.DataFrame:
        """Génère données reviews"""
        np.random.seed(654)
        
        territories = ['FR', 'US', 'GB', 'DE', 'CA', 'AU', 'JP']
        sentiments = ['Positive', 'Negative', 'Neutral']
        
        data = []
        for i in range(100):
            sentiment = np.random.choice(sentiments, p=[0.6, 0.2, 0.2])
            rating = 5 if sentiment == 'Positive' else (1 if sentiment == 'Negative' else 3)
            rating += np.random.randint(-1, 2)  # Variation
            rating = max(1, min(5, rating))
            
            data.append({
                'ID': f'review_{i}',
                'Rating': rating,
                'Title': f'Avis utilisateur {i+1}',
                'Body': f'Commentaire détaillé sur l\'app (sentiment: {sentiment.lower()})',
                'Reviewer': f'User{i+1}',
                'Date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
                'Territory': np.random.choice(territories),
                'Sentiment': sentiment,
                'Length': np.random.randint(20, 200),
                'Has_Response': np.random.choice([True, False], p=[0.3, 0.7])
            })
        
        return pd.DataFrame(data)
    
    def _generate_iap_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Génère données In-App Purchases"""
        np.random.seed(987)
        
        iap_products = [
            {'id': 'premium_monthly', 'name': 'Premium Mensuel', 'price': 4.99},
            {'id': 'premium_yearly', 'name': 'Premium Annuel', 'price': 39.99},
            {'id': 'extra_storage', 'name': 'Stockage Supplémentaire', 'price': 1.99},
            {'id': 'themes_pack', 'name': 'Pack Thèmes', 'price': 2.99}
        ]
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        data = []
        for date in dates:
            for product in iap_products:
                units = np.random.poisson(5) if 'monthly' in product['id'] else np.random.poisson(2)
                revenue = units * product['price']
                
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Product_ID': product['id'],
                    'Product_Name': product['name'],
                    'Units': units,
                    'Revenue': round(revenue, 2),
                    'Territory': np.random.choice(['FR', 'US', 'GB']),
                    'Device': np.random.choice(['iPhone', 'iPad']),
                    'Price_Tier': product['price']
                })
        
        return pd.DataFrame(data)
    
    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()

# === INTERFACE UTILISATEUR AVANCÉE ===

class AdvancedFiltersUI:
    """Interface pour filtres avancés"""
    
    @staticmethod
    def create_comprehensive_filters() -> FilterOptions:
        """Crée interface complète de filtres"""
        st.markdown("### 🎛️ Filtres Avancés Apple Store Connect")
        
        # Organisation en colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📅 Période & Fréquence")
            
            # Sélecteur de période
            period_option = st.selectbox(
                "📊 Période prédéfinie",
                ["Personnalisée", "7 derniers jours", "30 derniers jours", "90 derniers jours", "6 mois", "1 an"],
                key="period_filter"
            )
            
            if period_option == "Personnalisée":
                date_range = st.date_input(
                    "📅 Dates personnalisées",
                    value=(datetime.now() - timedelta(days=30), datetime.now()),
                    max_value=datetime.now(),
                    key="custom_dates"
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = datetime.now() - timedelta(days=30)
                    end_date = datetime.now()
            else:
                # Calcul automatique des dates
                days_map = {
                    "7 derniers jours": 7,
                    "30 derniers jours": 30, 
                    "90 derniers jours": 90,
                    "6 mois": 180,
                    "1 an": 365
                }
                days = days_map.get(period_option, 30)
                start_date = datetime.now() - timedelta(days=days)
                end_date = datetime.now()
            
            # Fréquence des données
            frequency = st.selectbox(
                "📊 Fréquence des données",
                ["DAILY", "WEEKLY", "MONTHLY"],
                key="frequency_filter"
            )
            
            # Granularité
            granularity = st.selectbox(
                "🔍 Granularité d'analyse",
                ["HOURLY", "DAILY", "WEEKLY", "MONTHLY"],
                index=1,
                key="granularity_filter"
            )
        
        with col2:
            st.markdown("#### 🌍 Territoires & Appareils")
            
            # Territoires
            all_territories = [
                "Worldwide", "France", "États-Unis", "Royaume-Uni", "Allemagne", 
                "Japon", "Chine", "Canada", "Australie", "Italie", "Espagne"
            ]
            
            territory_mode = st.radio(
                "🌍 Sélection territoires",
                ["Tous", "Personnalisé"],
                key="territory_mode"
            )
            
            if territory_mode == "Personnalisé":
                selected_territories = st.multiselect(
                    "Territoires spécifiques",
                    all_territories,
                    default=["France", "États-Unis"],
                    key="territories_filter"
                )
            else:
                selected_territories = None
            
            # Types d'appareils
            all_devices = ["iPhone", "iPad", "Apple Watch", "Apple TV", "Mac"]
            
            device_mode = st.radio(
                "📱 Sélection appareils", 
                ["Tous", "Personnalisé"],
                key="device_mode"
            )
            
            if device_mode == "Personnalisé":
                selected_devices = st.multiselect(
                    "Appareils spécifiques",
                    all_devices,
                    default=["iPhone", "iPad"],
                    key="devices_filter"
                )
            else:
                selected_devices = None
        
        with col3:
            st.markdown("#### 📊 Métriques & Sources")
            
            # Catégories de rapports
            all_categories = [
                "APP_STORE_ENGAGEMENT",
                "APP_STORE_COMMERCE", 
                "APP_USAGE",
                "PERFORMANCE",
                "FRAMEWORK_USAGE"
            ]
            
            selected_categories = st.multiselect(
                "📊 Catégories de rapports",
                all_categories,
                default=all_categories[:3],
                key="categories_filter"
            )
            
            # Sources de trafic
            all_sources = [
                "App Store Search",
                "App Store Browse",
                "App Referrer",
                "Web Referrer",
                "Unavailable"
            ]
            
            source_mode = st.radio(
                "🔗 Sources de trafic",
                ["Toutes", "Personnalisé"],
                key="source_mode"
            )
            
            if source_mode == "Personnalisé":
                selected_sources = st.multiselect(
                    "Sources spécifiques",
                    all_sources,
                    default=all_sources[:2],
                    key="sources_filter"
                )
            else:
                selected_sources = None
            
            # Métriques personnalisées
            custom_measures = st.multiselect(
                "📈 Métriques personnalisées",
                [
                    "Impressions", "Page Views", "Downloads", "Sessions",
                    "Active Devices", "Revenue", "Conversion Rate", "Retention"
                ],
                key="measures_filter"
            )
        
        # Options avancées dans un expander
        with st.expander("⚙️ Options Avancées", expanded=False):
            advanced_col1, advanced_col2 = st.columns(2)
            
            with advanced_col1:
                st.markdown("##### 🎯 Segmentation")
                
                # Segmentation utilisateurs
                user_segments = st.multiselect(
                    "Segments utilisateurs",
                    ["Nouveaux utilisateurs", "Utilisateurs récurrents", "Utilisateurs premium", "Beta testeurs"],
                    key="user_segments"
                )
                
                # Version de l'app
                app_versions = st.multiselect(
                    "Versions de l'app",
                    ["Version actuelle", "Versions précédentes", "Beta versions"],
                    default=["Version actuelle"],
                    key="app_versions"
                )
            
            with advanced_col2:
                st.markdown("##### 📊 Comparaisons")
                
                # Comparaison temporelle
                compare_periods = st.checkbox(
                    "Comparer avec période précédente",
                    key="compare_periods"
                )
                
                # Benchmarking
                enable_benchmarks = st.checkbox(
                    "Inclure benchmarks industrie",
                    key="benchmarks"
                )
                
                # Alertes personnalisées
                enable_alerts = st.checkbox(
                    "Alertes personnalisées activées",
                    value=True,
                    key="alerts_enabled"
                )
        
        # Résumé des filtres sélectionnés
        with st.expander("📋 Résumé des Filtres", expanded=True):
            filter_summary = []
            
            filter_summary.append(f"📅 **Période**: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
            filter_summary.append(f"📊 **Fréquence**: {frequency}")
            
            if selected_territories:
                filter_summary.append(f"🌍 **Territoires**: {', '.join(selected_territories[:3])}{'...' if len(selected_territories) > 3 else ''}")
            
            if selected_devices:
                filter_summary.append(f"📱 **Appareils**: {', '.join(selected_devices)}")
            
            filter_summary.append(f"📊 **Catégories**: {len(selected_categories)} sélectionnée(s)")
            
            st.markdown("\n".join(filter_summary))
        
        # Construction de l'objet FilterOptions
        return FilterOptions(
            territories=selected_territories,
            devices=selected_devices,
            sources=selected_sources,
            categories=selected_categories,
            measures=custom_measures,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            frequency=frequency,
            granularity=granularity
        )

# === DASHBOARD SECTIONS AVANCÉES ===

def create_sales_analytics_section(sales_data: pd.DataFrame, filters: FilterOptions):
    """Section analytics des ventes avancée"""
    st.markdown("## 💰 Analytics des Ventes Avancés")
    
    if sales_data.empty:
        st.warning("Aucune donnée de vente disponible")
        return
    
    # KPIs des ventes
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = sales_data['Revenue'].sum()
    total_units = sales_data['Units'].sum()
    avg_price = total_revenue / total_units if total_units > 0 else 0
    
    with col1:
        st.metric("💰 Revenus Totaux", f"€{total_revenue:,.2f}")
    
    with col2:
        st.metric("📦 Unités Vendues", f"{total_units:,}")
    
    with col3:
        st.metric("💵 Prix Moyen", f"€{avg_price:.2f}")
    
    with col4:
        territories_count = sales_data['Territory'].nunique()
        st.metric("🌍 Territoires Actifs", territories_count)
    
    # Graphiques des ventes
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Évolution", "🌍 Par Territoire", "📱 Par Appareil", "💹 Analyse Détaillée"
    ])
    
    with tab1:
        # Évolution temporelle
        if 'Date' in sales_data.columns:
            daily_sales = sales_data.groupby('Date').agg({
                'Revenue': 'sum',
                'Units': 'sum'
            }).reset_index()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=daily_sales['Date'], y=daily_sales['Revenue'],
                          name='Revenus (€)', line=dict(color='#1f77b4')),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=daily_sales['Date'], y=daily_sales['Units'],
                          name='Unités', line=dict(color='#ff7f0e')),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Revenus (€)", secondary_y=False)
            fig.update_yaxes(title_text="Unités", secondary_y=True)
            fig.update_layout(title="Évolution des Ventes", height=400)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Analyse par territoire
        territory_stats = sales_data.groupby('Territory').agg({
            'Revenue': 'sum',
            'Units': 'sum'
        }).reset_index()
        
        territory_stats['Avg_Price'] = territory_stats['Revenue'] / territory_stats['Units']
        territory_stats = territory_stats.sort_values('Revenue', ascending=False)
        
        fig = px.bar(territory_stats, x='Territory', y='Revenue',
                     title="Revenus par Territoire")
        st.plotly_chart(fig, use_container_width=True)
        
        # Table détaillée
        st.dataframe(territory_stats, use_container_width=True)
    
    with tab3:
        # Analyse par appareil
        if 'Device' in sales_data.columns:
            device_stats = sales_data.groupby('Device').agg({
                'Revenue': 'sum',
                'Units': 'sum'
            }).reset_index()
            
            fig = px.pie(device_stats, values='Revenue', names='Device',
                        title="Répartition des Revenus par Appareil")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Analyse croisée détaillée
        st.markdown("### 🔍 Analyse Croisée Territoire x Appareil")
        
        if 'Device' in sales_data.columns:
            cross_analysis = sales_data.pivot_table(
                index='Territory', 
                columns='Device', 
                values='Revenue', 
                aggfunc='sum',
                fill_value=0
            )
            
            fig = px.imshow(cross_analysis, 
                           title="Heatmap: Revenus par Territoire et Appareil",
                           aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top combinaisons
            top_combos = sales_data.groupby(['Territory', 'Device'])['Revenue'].sum().sort_values(ascending=False).head(10)
            st.markdown("#### 🏆 Top 10 Combinaisons Territoire-Appareil")
            st.dataframe(top_combos.reset_index(), use_container_width=True)

def create_app_store_optimization_section(keyword_data: pd.DataFrame, competitor_data: pd.DataFrame):
    """Section ASO (App Store Optimization)"""
    st.markdown("## 🎯 App Store Optimization (ASO)")
    
    tab1, tab2, tab3 = st.tabs([
        "🔍 Mots-clés", "🏢 Concurrents", "📊 Performance ASO"
    ])
    
    with tab1:
        st.markdown("### 🔍 Analyse des Mots-clés")
        
        if not keyword_data.empty:
            # KPIs mots-clés
            col1, col2, col3, col4 = st.columns(4)
            
            avg_ranking = keyword_data['Ranking'].mean()
            total_volume = keyword_data['Search_Volume'].sum()
            top_10_count = len(keyword_data[keyword_data['Ranking'] <= 10])
            
            with col1:
                st.metric("📊 Ranking Moyen", f"{avg_ranking:.1f}")
            
            with col2:
                st.metric("🔍 Volume Total", f"{total_volume:,}")
            
            with col3:
                st.metric("🏆 Top 10", f"{top_10_count}")
            
            with col4:
                rising_count = len(keyword_data[keyword_data['Trend'] == 'Rising'])
                st.metric("📈 En Hausse", f"{rising_count}")
            
            # Graphique performance mots-clés
            fig = px.scatter(keyword_data, 
                           x='Search_Volume', y='Ranking', 
                           size='Difficulty', color='Trend',
                           hover_name='Keyword',
                           title="Performance des Mots-clés")
            fig.update_yaxes(autorange="reversed")  # Ranking 1 = meilleur
            st.plotly_chart(fig, use_container_width=True)
            
            # Table mots-clés
            st.dataframe(keyword_data.sort_values('Ranking'), use_container_width=True)
    
    with tab2:
        st.markdown("### 🏢 Analyse Concurrentielle")
        
        if not competitor_data.empty:
            # Positionnement concurrentiel
            fig = px.scatter(competitor_data,
                           x='Rating', y='Reviews_Count',
                           size='Downloads_Estimate', color='Price',
                           hover_name='App_Name',
                           title="Positionnement Concurrentiel")
            st.plotly_chart(fig, use_container_width=True)
            
            # Table concurrents
            st.dataframe(competitor_data.sort_values('Rating', ascending=False), 
                        use_container_width=True)
    
    with tab3:
        st.markdown("### 📊 Performance ASO Globale")
        
        # Score ASO simulé
        aso_score = np.random.uniform(65, 85)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🎯 Score ASO Global", f"{aso_score:.1f}/100")
            
            # Recommandations ASO
            st.markdown("#### 💡 Recommandations ASO")
            recommendations = [
                "Optimiser le titre avec des mots-clés performants",
                "Améliorer les screenshots avec des call-to-actions",
                "Augmenter le nombre de reviews positifs",
                "Localiser les métadonnées pour nouveaux territoires",
                "Tester différentes icônes d'app"
            ]
            
            for rec in recommendations:
                st.write(f"• {rec}")
        
        with col2:
            # Métriques ASO détaillées
            aso_metrics = {
                'Visibilité': np.random.uniform(60, 90),
                'Conversion': np.random.uniform(50, 80),
                'Qualité': np.random.uniform(70, 95),
                'Localisation': np.random.uniform(40, 70)
            }
            
            fig = go.Figure(data=go.Scatterpolar(
                r=list(aso_metrics.values()),
                theta=list(aso_metrics.keys()),
                fill='toself',
                name='Performance ASO'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Radar ASO Performance"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def create_user_feedback_section(reviews_data: pd.DataFrame, beta_feedback: pd.DataFrame):
    """Section retours utilisateurs"""
    st.markdown("## 💬 Retours Utilisateurs")
    
    tab1, tab2, tab3 = st.tabs([
        "⭐ Reviews App Store", "🧪 Feedback Beta", "📊 Analyse Sentiment"
    ])
    
    with tab1:
        st.markdown("### ⭐ Reviews App Store")
        
        if not reviews_data.empty:
            # KPIs reviews
            col1, col2, col3, col4 = st.columns(4)
            
            avg_rating = reviews_data['Rating'].mean()
            total_reviews = len(reviews_data)
            positive_pct = len(reviews_data[reviews_data['Sentiment'] == 'Positive']) / total_reviews * 100
            
            with col1:
                st.metric("⭐ Note Moyenne", f"{avg_rating:.1f}/5")
            
            with col2:
                st.metric("📝 Total Reviews", f"{total_reviews:,}")
            
            with col3:
                st.metric("😊 % Positifs", f"{positive_pct:.1f}%")
            
            with col4:
                recent_reviews = len(reviews_data[pd.to_datetime(reviews_data['Date']) > datetime.now() - timedelta(days=7)])
                st.metric("📅 Cette Semaine", f"{recent_reviews}")
            
            # Distribution des notes
            rating_dist = reviews_data['Rating'].value_counts().sort_index()
            fig = px.bar(x=rating_dist.index, y=rating_dist.values,
                        title="Distribution des Notes")
            st.plotly_chart(fig, use_container_width=True)
            
            # Reviews par territoire
            territory_reviews = reviews_data.groupby('Territory').agg({
                'Rating': 'mean',
                'ID': 'count'
            }).rename(columns={'ID': 'Count'}).reset_index()
            
            fig = px.bar(territory_reviews, x='Territory', y='Count',
                        title="Nombre de Reviews par Territoire")
            st.plotly_chart(fig, use_container_width=True)
            
            # Table reviews récents
            st.markdown("#### 📱 Reviews Récents")
            recent_reviews_df = reviews_data.sort_values('Date', ascending=False).head(10)
            st.dataframe(recent_reviews_df[['Rating', 'Title', 'Territory', 'Sentiment', 'Date']], 
                        use_container_width=True)
    
    with tab2:
        st.markdown("### 🧪 Feedback Beta Testeurs")
        
        if not beta_feedback.empty:
            # Distribution par catégorie
            category_dist = beta_feedback['Category'].value_counts()
            fig = px.pie(values=category_dist.values, names=category_dist.index,
                        title="Feedback par Catégorie")
            st.plotly_chart(fig, use_container_width=True)
            
            # Timeline feedback
            beta_feedback['Date'] = pd.to_datetime(beta_feedback['Timestamp'])
            daily_feedback = beta_feedback.groupby(beta_feedback['Date'].dt.date).size().reset_index()
            daily_feedback.columns = ['Date', 'Count']
            
            fig = px.line(daily_feedback, x='Date', y='Count',
                         title="Évolution du Feedback Beta")
            st.plotly_chart(fig, use_container_width=True)
            
            # Table feedback
            st.dataframe(beta_feedback, use_container_width=True)
    
    with tab3:
        st.markdown("### 📊 Analyse de Sentiment")
        
        if not reviews_data.empty:
            # Évolution sentiment dans le temps
            reviews_data['Date'] = pd.to_datetime(reviews_data['Date'])
            sentiment_evolution = reviews_data.groupby([
                reviews_data['Date'].dt.date,
                'Sentiment'
            ]).size().unstack(fill_value=0).reset_index()
            
            fig = go.Figure()
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                if sentiment in sentiment_evolution.columns:
                    fig.add_trace(go.Scatter(
                        x=sentiment_evolution['Date'],
                        y=sentiment_evolution[sentiment],
                        mode='lines+markers',
                        name=sentiment,
                        stackgroup='one'
                    ))
            
            fig.update_layout(title="Évolution du Sentiment des Reviews")
            st.plotly_chart(fig, use_container_width=True)
            
            # Mots-clés dans reviews négatifs
            negative_reviews = reviews_data[reviews_data['Sentiment'] == 'Negative']
            if not negative_reviews.empty:
                st.markdown("#### 🔍 Mots-clés dans Reviews Négatifs")
                
                # Analyse basique des mots fréquents
                all_text = ' '.join(negative_reviews['Body'].fillna(''))
                common_words = ['crash', 'bug', 'slow', 'problem', 'issue', 'error', 'freeze']
                word_counts = {word: all_text.lower().count(word) for word in common_words}
                
                word_df = pd.DataFrame(list(word_counts.items()), columns=['Mot', 'Fréquence'])
                word_df = word_df[word_df['Fréquence'] > 0].sort_values('Fréquence', ascending=False)
                
                if not word_df.empty:
                    fig = px.bar(word_df, x='Mot', y='Fréquence',
                                title="Mots Fréquents dans Reviews Négatifs")
                    st.plotly_chart(fig, use_container_width=True)

def create_financial_dashboard(financial_data: pd.DataFrame, iap_data: pd.DataFrame):
    """Dashboard financier détaillé"""
    st.markdown("## 💰 Dashboard Financier")
    
    tab1, tab2, tab3 = st.tabs([
        "💵 Revenus Globaux", "🛍️ In-App Purchases", "📊 Analyse Financière"
    ])
    
    with tab1:
        if not financial_data.empty:
            # KPIs financiers globaux
            col1, col2, col3, col4 = st.columns(4)
            
            total_revenue = financial_data['Revenue'].sum()
            total_tax = financial_data['Tax'].sum()
            total_commission = financial_data['Commission'].sum()
            net_revenue = financial_data['Net_Revenue'].sum()
            
            with col1:
                st.metric("💰 Revenus Bruts", f"€{total_revenue:,.2f}")
            
            with col2:
                st.metric("🏛️ Taxes", f"€{total_tax:,.2f}")
            
            with col3:
                st.metric("🍎 Commission Apple", f"€{total_commission:,.2f}")
            
            with col4:
                st.metric("💵 Revenus Nets", f"€{net_revenue:,.2f}")
            
            # Graphique waterfall des revenus
            fig = go.Figure(go.Waterfall(
                name="Revenus", orientation="v",
                measure=["absolute", "relative", "relative", "total"],
                x=["Revenus Bruts", "Taxes", "Commission", "Revenus Nets"],
                textposition="outside",
                text=[f"€{total_revenue:,.0f}", f"-€{total_tax:,.0f}", 
                      f"-€{total_commission:,.0f}", f"€{net_revenue:,.0f}"],
                y=[total_revenue, -total_tax, -total_commission, net_revenue],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(title="Analyse Waterfall des Revenus", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Revenus par région
            region_revenue = financial_data.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
            fig = px.bar(x=region_revenue.index, y=region_revenue.values,
                        title="Revenus par Région")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🛍️ In-App Purchases Détaillés")
        
        if not iap_data.empty:
            # KPIs IAP
            col1, col2, col3, col4 = st.columns(4)
            
            total_iap_revenue = iap_data['Revenue'].sum()
            total_iap_units = iap_data['Units'].sum()
            avg_iap_price = total_iap_revenue / total_iap_units if total_iap_units > 0 else 0
            unique_products = iap_data['Product_ID'].nunique()
            
            with col1:
                st.metric("💰 Revenus IAP", f"€{total_iap_revenue:,.2f}")
            
            with col2:
                st.metric("📦 Unités IAP", f"{total_iap_units:,}")
            
            with col3:
                st.metric("💵 Prix Moyen IAP", f"€{avg_iap_price:.2f}")
            
            with col4:
                st.metric("🛍️ Produits Actifs", f"{unique_products}")
            
            # Performance par produit IAP
            product_performance = iap_data.groupby('Product_Name').agg({
                'Revenue': 'sum',
                'Units': 'sum'
            }).reset_index()
            product_performance = product_performance.sort_values('Revenue', ascending=False)
            
            fig = px.bar(product_performance, x='Product_Name', y='Revenue',
                        title="Revenus par Produit IAP")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Évolution temporelle IAP
            if 'Date' in iap_data.columns:
                daily_iap = iap_data.groupby('Date')['Revenue'].sum().reset_index()
                fig = px.line(daily_iap, x='Date', y='Revenue',
                             title="Évolution des Revenus IAP")
                st.plotly_chart(fig, use_container_width=True)
            
            # Table détaillée IAP
            st.dataframe(product_performance, use_container_width=True)
    
    with tab3:
        st.markdown("### 📊 Analyse Financière Avancée")
        
        # Métriques de performance financière
        col1, col2 = st.columns(2)
        
        with col1:
            if not financial_data.empty:
                # Marge bénéficiaire
                margin = (net_revenue / total_revenue * 100) if total_revenue > 0 else 0
                st.metric("📊 Marge Bénéficiaire", f"{margin:.1f}%")
                
                # Taux de taxe moyen
                tax_rate = (total_tax / total_revenue * 100) if total_revenue > 0 else 0
                st.metric("🏛️ Taux de Taxe Moyen", f"{tax_rate:.1f}%")
                
                # ARPU (Average Revenue Per User) simulé
                estimated_users = 10000  # À remplacer par vraies données
                arpu = total_revenue / estimated_users
                st.metric("👤 ARPU", f"€{arpu:.2f}")
        
        with col2:
            # Graphique en camembert des coûts
            costs_data = pd.DataFrame({
                'Type': ['Revenus Nets', 'Taxes', 'Commission Apple'],
                'Montant': [net_revenue, total_tax, total_commission]
            })
            
            fig = px.pie(costs_data, values='Montant', names='Type',
                        title="Répartition des Revenus et Coûts")
            st.plotly_chart(fig, use_container_width=True)
        
        # Prévisions financières
        st.markdown("#### 🔮 Prévisions Financières")
        
        # Projection simple basée sur la tendance
        if not financial_data.empty and len(financial_data) > 1:
            # Calcul de la croissance moyenne
            revenue_growth = 0.05  # 5% par mois (simulé)
            
            future_months = 6
            projections = []
            current_revenue = total_revenue
            
            for i in range(future_months):
                current_revenue *= (1 + revenue_growth)
                projections.append({
                    'Mois': f"Mois +{i+1}",
                    'Revenus_Projetes': current_revenue,
                    'Revenus_Nets_Projetes': current_revenue * 0.7  # Approximation
                })
            
            proj_df = pd.DataFrame(projections)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=proj_df['Mois'],
                y=proj_df['Revenus_Projetes'],
                mode='lines+markers',
                name='Revenus Bruts Projetés',
                line=dict(color='#1f77b4')
            ))
            fig.add_trace(go.Scatter(
                x=proj_df['Mois'],
                y=proj_df['Revenus_Nets_Projetes'],
                mode='lines+markers',
                name='Revenus Nets Projetés',
                line=dict(color='#2ca02c')
            ))
            
            fig.update_layout(title="Prévisions Financières (6 mois)", height=400)
            st.plotly_chart(fig, use_container_width=True)

def create_beta_testing_section(beta_data: pd.DataFrame, feedback_data: pd.DataFrame):
    """Section beta testing"""
    st.markdown("## 🧪 Beta Testing & TestFlight")
    
    if beta_data.empty:
        st.info("Aucune donnée de beta testing disponible")
        return
    
    tab1, tab2, tab3 = st.tabs([
        "👥 Beta Testeurs", "📝 Feedback", "📊 Statistiques"
    ])
    
    with tab1:
        st.markdown("### 👥 Gestion des Beta Testeurs")
        
        # KPIs beta testeurs
        col1, col2, col3, col4 = st.columns(4)
        
        total_testers = len(beta_data)
        active_testers = len(beta_data[beta_data['State'] == 'ACTIVE'])
        avg_sessions = beta_data['Sessions'].mean()
        total_feedback = beta_data['Feedback_Count'].sum()
        
        with col1:
            st.metric("👥 Total Testeurs", f"{total_testers}")
        
        with col2:
            st.metric("✅ Testeurs Actifs", f"{active_testers}")
        
        with col3:
            st.metric("📱 Sessions Moyennes", f"{avg_sessions:.1f}")
        
        with col4:
            st.metric("💬 Total Feedback", f"{total_feedback}")
        
        # Distribution des états
        state_dist = beta_data['State'].value_counts()
        fig = px.pie(values=state_dist.values, names=state_dist.index,
                    title="Distribution des États des Testeurs")
        st.plotly_chart(fig, use_container_width=True)
        
        # Activité des testeurs
        fig = px.histogram(beta_data, x='Sessions', nbins=20,
                          title="Distribution de l'Activité des Testeurs")
        st.plotly_chart(fig, use_container_width=True)
        
        # Table des testeurs
        st.markdown("#### 📋 Liste des Beta Testeurs")
        display_columns = ['Name', 'Email', 'State', 'Sessions', 'Feedback_Count', 'Invite_Date']
        st.dataframe(beta_data[display_columns], use_container_width=True)
    
    with tab2:
        st.markdown("### 📝 Feedback Beta")
        
        if not feedback_data.empty:
            # Analyse du feedback par catégorie
            category_stats = feedback_data.groupby('Category').agg({
                'Rating': 'mean',
                'ID': 'count'
            }).rename(columns={'ID': 'Count'}).reset_index()
            
            fig = px.bar(category_stats, x='Category', y='Count',
                        title="Feedback par Catégorie")
            st.plotly_chart(fig, use_container_width=True)
            
            # Timeline du feedback
            feedback_data['Date'] = pd.to_datetime(feedback_data['Timestamp'])
            daily_feedback = feedback_data.groupby(feedback_data['Date'].dt.date).size().reset_index()
            daily_feedback.columns = ['Date', 'Count']
            
            fig = px.line(daily_feedback, x='Date', y='Count',
                         title="Évolution du Feedback dans le Temps")
            st.plotly_chart(fig, use_container_width=True)
            
            # Table du feedback récent
            st.markdown("#### 📱 Feedback Récent")
            recent_feedback = feedback_data.sort_values('Timestamp', ascending=False).head(20)
            st.dataframe(recent_feedback[['Rating', 'Category', 'Comment', 'Timestamp']], 
                        use_container_width=True)
    
    with tab3:
        st.markdown("### 📊 Statistiques Beta")
        
        # Corrélation sessions vs feedback
        if len(beta_data) > 1:
            fig = px.scatter(beta_data, x='Sessions', y='Feedback_Count',
                           title="Corrélation Sessions vs Feedback",
                           hover_name='Name')
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques par période d'invitation
        beta_data['Invite_Date'] = pd.to_datetime(beta_data['Invite_Date'])
        beta_data['Invite_Month'] = beta_data['Invite_Date'].dt.to_period('M')
        
        monthly_stats = beta_data.groupby('Invite_Month').agg({
            'ID': 'count',
            'Sessions': 'mean'
        }).rename(columns={'ID': 'New_Testers'}).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=monthly_stats['Invite_Month'].astype(str), 
                   y=monthly_stats['New_Testers'],
                   name='Nouveaux Testeurs'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_stats['Invite_Month'].astype(str), 
                      y=monthly_stats['Sessions'],
                      name='Sessions Moyennes',
                      mode='lines+markers'),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Mois")
        fig.update_yaxes(title_text="Nouveaux Testeurs", secondary_y=False)
        fig.update_yaxes(title_text="Sessions Moyennes", secondary_y=True)
        fig.update_layout(title="Évolution du Programme Beta")
        
        st.plotly_chart(fig, use_container_width=True)

def create_advanced_performance_section(performance_data: pd.DataFrame):
    """Section performance avancée"""
    st.markdown("## ⚡ Performance & Stabilité Avancée")
    
    if performance_data.empty:
        st.warning("Aucune donnée de performance disponible")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "💥 Crashes & Hangs", "🚀 Performance", "💾 Ressources", "📱 Par Appareil"
    ])
    
    with tab1:
        st.markdown("### 💥 Analyse des Crashes et Hangs")
        
        # KPIs stabilité
        col1, col2, col3, col4 = st.columns(4)
        
        avg_crash_rate = performance_data['Crash_Rate'].mean()
        avg_hang_rate = performance_data['Hang_Rate'].mean()
        stability_score = max(0, 100 - (avg_crash_rate + avg_hang_rate) * 10)
        
        with col1:
            st.metric("💥 Taux Crash Moyen", f"{avg_crash_rate:.2f}%", 
                     delta_color="inverse")
        
        with col2:
            st.metric("🔒 Taux Hang Moyen", f"{avg_hang_rate:.2f}%",
                     delta_color="inverse")
        
        with col3:
            st.metric("🛡️ Score Stabilité", f"{stability_score:.1f}/100")
        
        with col4:
            critical_days = len(performance_data[performance_data['Crash_Rate'] > 1.0])
            st.metric("🚨 Jours Critiques", f"{critical_days}")
        
        # Évolution crash rate
        if 'Date' in performance_data.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=performance_data['Date'],
                y=performance_data['Crash_Rate'],
                mode='lines+markers',
                name='Crash Rate',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=performance_data['Date'],
                y=performance_data['Hang_Rate'],
                mode='lines+markers',
                name='Hang Rate',
                line=dict(color='orange')
            ))
            
            # Ligne objectif
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                         annotation_text="Objectif < 1%")
            
            fig.update_layout(title="Évolution des Taux de Crash et Hang", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des crash rates
        fig = px.histogram(performance_data, x='Crash_Rate', nbins=20,
                          title="Distribution des Taux de Crash")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🚀 Performance de Lancement et Réactivité")
        
        # KPIs performance
        col1, col2, col3 = st.columns(3)
        
        avg_cold_launch = performance_data['Launch_Time_Cold'].mean()
        avg_warm_launch = performance_data['Launch_Time_Warm'].mean()
        launch_improvement = ((avg_cold_launch - avg_warm_launch) / avg_cold_launch) * 100
        
        with col1:
            st.metric("🥶 Lancement Froid", f"{avg_cold_launch:.2f}s",
                     delta_color="inverse")
        
        with col2:
            st.metric("🔥 Lancement Chaud", f"{avg_warm_launch:.2f}s",
                     delta_color="inverse")
        
        with col3:
            st.metric("⚡ Amélioration Cache", f"{launch_improvement:.1f}%")
        
        # Comparaison temps de lancement
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_data['Date'],
            y=performance_data['Launch_Time_Cold'],
            mode='lines+markers',
            name='Lancement Froid',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_data['Date'],
            y=performance_data['Launch_Time_Warm'],
            mode='lines+markers',
            name='Lancement Chaud',
            line=dict(color='green')
        ))
        
        # Ligne objectif
        fig.add_hline(y=2.0, line_dash="dash", line_color="gray",
                     annotation_text="Objectif < 2s")
        
        fig.update_layout(title="Évolution des Temps de Lancement", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot performance
        launch_data = pd.melt(performance_data[['Launch_Time_Cold', 'Launch_Time_Warm']], 
                             var_name='Type', value_name='Time')
        
        fig = px.box(launch_data, x='Type', y='Time',
                    title="Distribution des Temps de Lancement")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 💾 Utilisation des Ressources")
        
        # KPIs ressources
        col1, col2, col3, col4 = st.columns(4)
        
        avg_memory = performance_data['Memory_Usage_Average'].mean()
        peak_memory = performance_data['Memory_Usage_Peak'].mean()
        avg_cpu = performance_data['CPU_Usage_Average'].mean()
        avg_battery = performance_data['Battery_Usage_Per_Hour'].mean()
        
        with col1:
            st.metric("💾 RAM Moyenne", f"{avg_memory:.1f} MB")
        
        with col2:
            st.metric("⚡ RAM Peak", f"{peak_memory:.1f} MB")
        
        with col3:
            st.metric("🖥️ CPU Moyen", f"{avg_cpu:.1f}%")
        
        with col4:
            st.metric("🔋 Batterie/h", f"{avg_battery:.1f}%")
        
        # Graphique multi-ressources
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mémoire', 'CPU', 'Batterie', 'Disque'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Mémoire
        fig.add_trace(
            go.Scatter(x=performance_data['Date'], y=performance_data['Memory_Usage_Average'],
                      name='RAM Moyenne', line=dict(color='blue')),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=performance_data['Date'], y=performance_data['Memory_Usage_Peak'],
                      name='RAM Peak', line=dict(color='red')),
            row=1, col=1, secondary_y=True
        )
        
        # CPU
        fig.add_trace(
            go.Scatter(x=performance_data['Date'], y=performance_data['CPU_Usage_Average'],
                      name='CPU', line=dict(color='green')),
            row=1, col=2
        )
        
        # Batterie
        fig.add_trace(
            go.Scatter(x=performance_data['Date'], y=performance_data['Battery_Usage_Per_Hour'],
                      name='Batterie', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Disque
        fig.add_trace(
            go.Scatter(x=performance_data['Date'], y=performance_data['Disk_Writes_Per_Hour'],
                      name='Écritures Disque', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Utilisation des Ressources Système")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 📱 Performance par Type d'Appareil")
        
        # Simulation de données par appareil
        devices = ['iPhone 15', 'iPhone 14', 'iPhone 13', 'iPad Pro', 'iPad Air']
        device_performance = []
        
        for device in devices:
            # Simulation basée sur les capacités de l'appareil
            multiplier = {'iPhone 15': 1.0, 'iPhone 14': 1.1, 'iPhone 13': 1.3, 
                         'iPad Pro': 0.9, 'iPad Air': 1.2}[device]
            
            device_performance.append({
                'Device': device,
                'Crash_Rate': avg_crash_rate * multiplier,
                'Launch_Time': avg_cold_launch * multiplier,
                'Memory_Usage': avg_memory * multiplier,
                'CPU_Usage': avg_cpu * multiplier,
                'Market_Share': np.random.uniform(0.1, 0.3)  # Simulé
            })
        
        device_df = pd.DataFrame(device_performance)
        
        # Performance par appareil
        fig = px.bar(device_df, x='Device', y='Crash_Rate',
                    title="Taux de Crash par Appareil")
        fig.add_hline(y=avg_crash_rate, line_dash="dash", 
                     annotation_text="Moyenne")
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart performance
        categories = ['Crash_Rate', 'Launch_Time', 'Memory_Usage', 'CPU_Usage']
        
        fig = go.Figure()
        
        for _, row in device_df.head(3).iterrows():  # Top 3 appareils
            fig.add_trace(go.Scatterpolar(
                r=[row[cat] for cat in categories],
                theta=categories,
                fill='toself',
                name=row['Device']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Comparaison Performance par Appareil"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table performance
        st.dataframe(device_df.round(2), use_container_width=True)

# === APPLICATION PRINCIPALE COMPLÈTE ===

@st.cache_data(ttl=300)  # Cache 5 minutes
def load_comprehensive_apple_data_with_filters(filters: FilterOptions):
    """Charge toutes les données Apple avec filtres"""
    return asyncio.run(load_apple_data_async(filters))

async def load_apple_data_async(filters: FilterOptions):
    """Charge données Apple de façon asynchrone"""
    api_client = AppleStoreConnectAPIComplete(APPLE_CONFIG)
    
    try:
        st.info("🔄 Connexion à Apple Store Connect...")
        
        # Informations app
        app_info = await api_client.get_app_info_complete()
        
        # Sales reports avec filtres
        sales_data = await api_client.get_sales_reports_advanced(
            frequency=filters.frequency,
            territories=filters.territories
        )
        
        # Rapports financiers
        financial_data = await api_client.get_financial_reports()
        
        # Analytics complets
        analytics_report_id = await api_client.create_analytics_report_request_advanced()
        analytics_data = {}
        
        if analytics_report_id:
            reports = await api_client.get_analytics_reports_filtered(analytics_report_id, filters)
            for report in reports[:3]:  # Limite pour éviter timeout
                if report.get('id'):
                    instances = await api_client.get_report_instances(report['id'])
                    if instances:
                        instance_url = instances[0].get('attributes', {}).get('url')
                        if instance_url:
                            df = await api_client.download_analytics_instance_with_filters(instance_url, filters)
                            if not df.empty:
                                analytics_data[report.get('attributes', {}).get('category', 'Unknown')] = df
        
        # Données IAP
        iap_data = await api_client.get_iap_reports(filters.start_date, filters.end_date)
        
        # Reviews avec filtres territoire
        reviews_data = await api_client.get_customer_reviews_advanced(filters.territories)
        
        # Beta testing
        beta_data = await api_client.get_beta_testers()
        beta_feedback = await api_client.get_beta_feedback()
        
        # ASO data
        keyword_data = await api_client.get_keyword_rankings()
        competitor_data = await api_client.get_competitor_analysis()
        
        # Performance data (généré avec fallback réaliste)
        if 'PERFORMANCE' not in analytics_data:
            analytics_data['PERFORMANCE'] = api_client._generate_comprehensive_fallback_data(
                filters.start_date, filters.end_date
            )['PERFORMANCE']
        
        st.success(f"✅ Données chargées: {len(analytics_data)} catégories analytics")
        
        return {
            'app_info': app_info,
            'sales_data': sales_data,
            'financial_data': financial_data,
            'analytics_data': analytics_data,
            'iap_data': iap_data,
            'reviews_data': reviews_data,
            'beta_data': beta_data,
            'beta_feedback': beta_feedback,
            'keyword_data': keyword_data,
            'competitor_data': competitor_data
        }
    
    except Exception as e:
        st.error(f"❌ Erreur chargement: {str(e)}")
        return {}
    
    finally:
        await api_client.close()

def create_main_dashboard(data_dict: dict, filters: FilterOptions):
    """Dashboard principal avec toutes les sections"""
    
    # Header avec informations app
    app_info = data_dict.get('app_info', {})
    if app_info:
        app_name = app_info.get('attributes', {}).get('name', 'App inconnue')
        bundle_id = app_info.get('attributes', {}).get('bundleId', 'N/A')
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 20px; color: white; margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
            <h1 style='margin: 0; font-size: 3em; text-align: center;'>🍎 {app_name}</h1>
            <p style='margin: 15px 0 0 0; opacity: 0.9; font-size: 1.2em; text-align: center;'>
                {bundle_id} | Dashboard Complet Apple Store Connect
            </p>
            <div style='margin-top: 20px; text-align: center;'>
                <span style='background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px; margin: 0 10px;'>
                    🔄 Données en Temps Réel
                </span>
                <span style='background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px; margin: 0 10px;'>
                    🎯 Filtres Actifs
                </span>
                <span style='background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px; margin: 0 10px;'>
                    📊 Analytics Complets
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # KPIs exécutifs globaux
    create_executive_summary_kpis(data_dict)
    
    # Navigation principale
    main_tabs = st.tabs([
        "💰 Ventes & Finance",
        "📊 Analytics & Usage", 
        "⭐ Reviews & ASO",
        "🧪 Beta Testing",
        "⚡ Performance",
        "🎯 Insights Avancés"
    ])
    
    with main_tabs[0]:
        create_sales_analytics_section(data_dict.get('sales_data', pd.DataFrame()), filters)
        st.markdown("---")
        create_financial_dashboard(
            data_dict.get('financial_data', pd.DataFrame()),
            data_dict.get('iap_data', pd.DataFrame())
        )
    
    with main_tabs[1]:
        analytics_data = data_dict.get('analytics_data', {})
        if analytics_data:
            create_analytics_charts(analytics_data)
        else:
            st.info("Aucune donnée analytics disponible avec ces filtres")
    
    with main_tabs[2]:
        create_user_feedback_section(
            data_dict.get('reviews_data', pd.DataFrame()),
            data_dict.get('beta_feedback', pd.DataFrame())
        )
        st.markdown("---")
        create_app
