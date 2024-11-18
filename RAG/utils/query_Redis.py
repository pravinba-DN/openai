import redis

def print_redis_data(redis_client):
    # Retrieve all keys from Redis
    keys = redis_client.keys('*')  # '*' matches all keys
    
    if not keys:
        print("No data found in Redis.")
        return
    
    print("Data stored in Redis:")

    for key in keys:
        # Decode the key (if needed) to a string
        key = key.decode('utf-8')
        
        # Check the type of the value associated with this key
        key_type = redis_client.type(key).decode('utf-8')
        
        print(f"Key: {key}, Type: {key_type}")
        
        # Based on the key type, retrieve and print the data
        if key_type == 'string':
            value = redis_client.get(key).decode('utf-8')
            print(f"  Value: {value}")
        elif key_type == 'list':
            value = redis_client.lrange(key, 0, -1)
            value = [v.decode('utf-8') for v in value]  # Decode the list elements
            print(f"  Value: {value}")
        elif key_type == 'set':
            value = redis_client.smembers(key)
            value = [v.decode('utf-8') for v in value]  # Decode the set elements
            print(f"  Value: {value}")
        elif key_type == 'hash':
            value = redis_client.hgetall(key)
            value = {k.decode('utf-8'): v.decode('utf-8') for k, v in value.items()}  # Decode hash keys and values
            print(f"  Value: {value}")
        elif key_type == 'zset':
            value = redis_client.zrange(key, 0, -1, withscores=True)
            value = [(v.decode('utf-8'), score) for v, score in value]  # Decode elements in sorted set
            print(f"  Value: {value}")
        else:
            print(f"  Unsupported data type: {key_type}")

if __name__ == "__main__":
    # Connect to the local Redis server (default host and port)
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=False)

    # Print all the data stored in Redis
    print_redis_data(redis_client)
