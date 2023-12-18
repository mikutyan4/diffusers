import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def scaled_dot_product_attention(query, key, scale=None):
    """
    Computes the scaled dot product attention.

    Parameters:
    - query: a tensor of shape [batch_size, query_length, hidden_dim]
    - key: a tensor of shape [batch_size, key_length, hidden_dim]
    - scale: a scaling factor (optional)

    Returns:
    - attention_weights: a tensor of shape [batch_size, query_length, key_length]
    """
    # Perform the dot product between query and key
    #only show the first head
    dot_product = torch.bmm(query[:,:,:,0], key.transpose(1, 2)[:,:,:,0])

    if scale is None:
        # If no scale provided, we default to the square root of the key dimension
        scale = torch.sqrt(torch.tensor(key.size(-1), dtype=dot_product.dtype, device=dot_product.device))

    # Scale the dot product
    scaled_dot_product = dot_product / scale

    # Apply softmax to get the attention weights
    attention_weights = F.softmax(scaled_dot_product, dim=-1)

    return attention_weights

def save_attention_map(attention_weights):
     # Get current date and time
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    plt.figure(figsize=(10,10))
    #we only show the first head
    sns.heatmap(attention_weights[0].detach().cpu().numpy(), cmap='viridis')
    plt.title(f'Attention Map of {timestamp_str}')
    # Save the figure with the timestamp in the filename
    plt.savefig(f'attention_map_{timestamp_str}.png')
    # Close the plot to free up memory
    plt.close()

def draw(query, key, scale=None):
     attention_weights = scaled_dot_product_attention(query, key, scale=None)
     save_attention_map(attention_weights)