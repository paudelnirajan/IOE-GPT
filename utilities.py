# Keywords that trigger checkpoint deletion
RESET_KEYWORDS = ["menu", "reload", "reset", "restart", "clear"]

def should_reset_checkpoint(query: str) -> bool:
    """Check if the query contains any reset keywords"""
    return any(keyword in query.lower() for keyword in RESET_KEYWORDS)

from langgraph.checkpoint.redis import RedisSaver

def delete_thread_checkpoints(redis_saver: RedisSaver, thread_id: str):
    """Delete all checkpoints for a specific thread ID
    
    Args:
        redis_saver: RedisSaver instance
        thread_id: The thread ID to delete checkpoints for
    """
    if redis_saver:
        try:
            # Get the Redis client from the saver
            redis_client = redis_saver._redis
            
            # Patterns to match all types of checkpoint keys for this thread
            patterns = [
                f"checkpoint:{thread_id}:__empty__:*",           # Main checkpoint
                f"checkpoint_write:{thread_id}:__empty__:*",     # Write checkpoints
                f"checkpoint_blob:{thread_id}:__empty__:*"       # Blob checkpoints
            ]
            
            all_keys = []
            # Find all matching keys for each pattern
            for pattern in patterns:
                keys = redis_client.keys(pattern)
                all_keys.extend(keys)
            
            if all_keys:
                # Delete all found keys
                redis_client.delete(*all_keys)
                print(f"[INFO] Deleted {len(all_keys)} checkpoints for thread_id: {thread_id}")
                # print(f"[INFO] Deleted keys: {all_keys}")
            else:
                print(f"[INFO] No checkpoints found for thread_id: {thread_id}")
                
        except Exception as e:
            print(f"[ERROR] Failed to delete checkpoints: {str(e)}")
            print(f"[ERROR] Thread ID: {thread_id}")