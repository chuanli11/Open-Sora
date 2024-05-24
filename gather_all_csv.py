import pandas as pd

# Manually define the list of CSV files to merge
# csv_files = [
#     '/home/ubuntu/ml-Illinois/chuan/datasets/animals/animal_universe_content/meta_clips_info_fmin1_aes_aesmin4.85_aio_cleaned2.csv',
#     '/home/ubuntu/ml-Illinois/chuan/datasets/animals/animal_8k_videos_ultra_hd/meta_clips_info_fmin1_aes_aesmin4.85_merged_success_clean2.csv',
#     '/home/ubuntu/ml-Illinois/chuan/datasets/animals/animal_8k_videos_ultra_hd_2/meta_clips_info_fmin1_aes_aesmin4.85_merged_success_clean2.csv',
#     '/home/ubuntu/ml-Illinois/chuan/datasets/animals/animal_8k_videos_ultra_hd/meta_clips_info_fmin1_aes_aesmin4.85_merged_failed_clean2.csv',
#     '/home/ubuntu/ml-Illinois/chuan/datasets/animals/animal_8k_videos_ultra_hd_2/meta_clips_info_fmin1_aes_aesmin4.85_merged_failed_clean2.csv',
# ]

# csv_files = [
#     '/home/ubuntu/ml-1cc/legos/jh_data/2024-05-16T19_30_42_vinfo_cleaned.csv',
#     '/home/ubuntu/ml-1cc/legos/mh-100-legoland-motion.csv',
# ]

# csv_files = [
#     '/home/ubuntu/ml-1cc/legos/lego_24k.csv',
#     '/home/ubuntu/ml-1cc/legos/lego-image-1-fixed.csv',
# ]


csv_files = [
    '/home/ubuntu/ml-1cc/legos/lego_24k.csv',
    '/home/ubuntu/ml-1cc/legos/lego-image-1-fixed.csv',
    '/home/ubuntu/ml-1cc/legos/image_Alexander_Studios/Alexander_Studios2024-05-21T22_33_53_info_noempty.csv',
    '/home/ubuntu/ml-1cc/legos/image_lego-music-playlist_PLr2ma_aujNX2SjalY8VAGe1q5iZR3KcvI/lego-music-playlist_PLr2ma_aujNX2SjalY8VAGe1q5iZR3KcvI2024-05-21T23_16_14_info_noempty_cleaned.csv',
]


# Read and concatenate all CSV files into one DataFrame
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('/home/ubuntu/ml-1cc/legos/lego_24k_15k_2k.csv', index=False)

# Print the number of rows in the merged DataFrame for verification
print(f"Number of rows in the merged file: {len(merged_df)}")
