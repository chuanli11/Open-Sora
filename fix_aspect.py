import pandas as pd

# Function to swap 'height' and 'width' values and set 'aspect_ratio' to 0.5625
def modify_csv(input_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(input_file_path)
    
    # Swap the values of 'height' and 'width'
    df['height'], df['width'] = df['width'], df['height']
    
    # Set all values in 'aspect_ratio' column to 0.5625
    df['aspect_ratio'] = 0.5625
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)
    print(f"Modified file has been saved as '{output_file_path}'.")

# Example usage
input_file_path = '/home/ubuntu/ml-1cc/legos/lego-image-1.csv'
output_file_path = '/home/ubuntu/ml-1cc/legos/lego-image-1-fixed.csv'
modify_csv(input_file_path, output_file_path)
