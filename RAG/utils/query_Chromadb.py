import chromadb

def fetch_collection_data(collection_ids=None):
    # Initialize the Chroma client
    client = chromadb.Client()

    # If collection_ids is empty or None, fetch all collections
    if not collection_ids:
        # Retrieve all collections
        collections = client.list_collections()
        collection_ids = [collection.id for collection in collections]

    # Loop through each collection ID and fetch its data
    collections_data = {}
    
    for collection_id in collection_ids:
        try:
            # Get the collection by ID
            collection = client.get_collection(collection_id)
            
            if collection:
                # Fetch collection data, for example, the list of documents
                collections_data[collection_id] = collection.get_all()
            else:
                print(f"Collection with ID {collection_id} not found.")
                collections_data[collection_id] = None
        except Exception as e:
            print(f"Error fetching collection {collection_id}: {e}")
            collections_data[collection_id] = None

    return collections_data


if __name__ == "__main__":
    # Example: if the list is empty, all collections will be fetched
    collection_ids = []  # Empty list to fetch all collections

    # Fetch and print the data from the collections
    collections_data = fetch_collection_data(collection_ids)
    
    for collection_id, data in collections_data.items():
        print(f"Data for collection {collection_id}: {data}")
