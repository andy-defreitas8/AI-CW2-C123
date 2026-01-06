
import csv
import argparse

def remove_label(input_path, output_path):
    with open(input_path, newline='', encoding='utf-8') as fin:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            print('Input file is empty.')
            return

        if 'label' not in header:
            print("No 'label' column found. Copying file to output.")
            with open(output_path, 'w', newline='', encoding='utf-8') as fout:
                writer = csv.writer(fout)
                writer.writerow(header)
                for row in reader:
                    writer.writerow(row)
            print(f'Wrote {output_path}')
            return

        idx = header.index('label')
        new_header = header[:idx] + header[idx+1:]

        with open(output_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            writer.writerow(new_header)
            for row in reader:
                # handle rows shorter than expected
                if len(row) <= idx:
                    writer.writerow(row)
                else:
                    new_row = row[:idx] + row[idx+1:]
                    writer.writerow(new_row)

    print(f"Removed 'label' and wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Remove 'label' column from CSV")
    parser.add_argument('-i', '--input', default='combined_landmarks_clean.csv', help='Input CSV path')
    parser.add_argument('-o', '--output', default='combined_landmarks_no_label.csv', help='Output CSV path')
    args = parser.parse_args()
    remove_label(args.input, args.output)


if __name__ == '__main__':
    main()
