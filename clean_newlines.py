import csv

def remove_newlines_from_csv(input_file, output_file):
    # Read the input CSV file
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    # Remove newlines from each cell
    cleaned_rows = []
    for row in rows:
        cleaned_row = [cell.replace('\n', ' ').replace('\r', ' ') for cell in row]
        cleaned_rows.append(cleaned_row)

    # Write the cleaned data to the output CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)

# Example usage
input_csv = '/home/ubuntu/ml-1cc/legos/image_lego-music-playlist_PLr2ma_aujNX2SjalY8VAGe1q5iZR3KcvI/lego-music-playlist_PLr2ma_aujNX2SjalY8VAGe1q5iZR3KcvI2024-05-21T23_16_14_info_noempty.csv'
output_csv = '/home/ubuntu/ml-1cc/legos/image_lego-music-playlist_PLr2ma_aujNX2SjalY8VAGe1q5iZR3KcvI/lego-music-playlist_PLr2ma_aujNX2SjalY8VAGe1q5iZR3KcvI2024-05-21T23_16_14_info_noempty_cleaned.csv'
remove_newlines_from_csv(input_csv, output_csv)
