import requests
import time
import json

def test_basic_connection():
    """Test basic Kraken connectivity"""
    print("Testing Kraken connection...")
    
    # Test 1: Basic ping
    try:
        response = requests.get("https://api.kraken.com/0/public/Time", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Basic connection works")
        else:
            print("❌ Connection failed")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Test 2: Get BTC price (simplest possible)
    try:
        response = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            
            if 'result' in data and data['result']:
                for pair, info in data['result'].items():
                    if 'c' in info:
                        price = float(info['c'][0])
                        print(f"✅ BTC Price: ${price:,.2f}")
                        return True
        
        print("❌ No price data received")
        return False
        
    except Exception as e:
        print(f"❌ Price fetch error: {e}")
        return False

def test_alternative_endpoints():
    """Test alternative ways to get BTC price"""
    
    endpoints = [
        "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        "https://api.kraken.com/0/public/Ticker?pair=XXBTZUSD", 
        "https://api.kraken.com/0/public/Ticker?pair=BTCUSD"
    ]
    
    for endpoint in endpoints:
        try:
            print(f"Trying: {endpoint}")
            response = requests.get(endpoint, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if not data.get('error') and data.get('result'):
                    for pair, info in data['result'].items():
                        if 'c' in info:
                            price = float(info['c'][0])
                            print(f"✅ SUCCESS: {endpoint} -> ${price:,.2f}")
                            return endpoint, price
            
            print(f"   Failed: {response.status_code}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    return None, None

def simple_price_fetcher():
    """Ultra-simple price fetcher that should work"""
    
    try:
        # Most basic request possible
        url = "https://api.kraken.com/0/public/Ticker"
        params = {"pair": "XBTUSD"}
        
        response = requests.get(url, params=params, timeout=20)
        
        print(f"URL: {response.url}")
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            text = response.text
            print(f"Raw response: {text[:200]}...")
            
            data = response.json()
            print(f"JSON data: {data}")
            
            # Extract price
            if 'result' in data:
                for key, value in data['result'].items():
                    if 'c' in value:
                        price = float(value['c'][0])
                        print(f"✅ PRICE FOUND: ${price:,.2f}")
                        return price
        
        return None
        
    except Exception as e:
        print(f"❌ Simple fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔍 KRAKEN CONNECTION DIAGNOSTIC")
    print("=" * 40)
    
    # Test 1: Basic connection
    if not test_basic_connection():
        print("\n❌ BASIC CONNECTION FAILED")
        print("Possible issues:")
        print("1. No internet connection")
        print("2. Firewall blocking requests") 
        print("3. Your ISP blocks Kraken")
        print("4. Kraken API is down")
        exit(1)
    
    # Test 2: Alternative endpoints
    print("\n🔄 Testing alternative endpoints...")
    working_endpoint, price = test_alternative_endpoints()
    
    if working_endpoint:
        print(f"\n✅ SOLUTION FOUND!")
        print(f"Working endpoint: {working_endpoint}")
        print(f"BTC Price: ${price:,.2f}")
    else:
        print("\n❌ NO WORKING ENDPOINTS")
        
        # Test 3: Ultra-simple fetch
        print("\n🔄 Trying ultra-simple fetch...")
        price = simple_price_fetcher()
        
        if not price:
            print("\n❌ ALL TESTS FAILED")
            print("\nTroubleshooting steps:")
            print("1. Check internet: ping google.com")
            print("2. Try different network (mobile hotspot)")
            print("3. Disable VPN if using one")
            print("4. Check firewall settings")
            print("5. Try from different location")
        else:
            print(f"\n✅ SUCCESS: ${price:,.2f}")