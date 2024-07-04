def compare_files(file1_path, file2_path):
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            line_num = 1
            lines_file1 = []
            lines_file2 = []

            while True:
                line1 = file1.readline()
                line2 = file2.readline()

                lines_file1.append(line1.strip())
                lines_file2.append(line2.strip())

                if len(lines_file1) > 10:
                    lines_file1.pop(0)
                    lines_file2.pop(0)

                if not line1 and not line2:
                    print("Files are identical")
                    break

                if line1 != line2:
                    print(f"Files differ at line {line_num}")
                    print("Leading up to the difference:")
                    for i in range(len(lines_file1) - 1):
                        print(f"Line {line_num - len(lines_file1) + i + 1}:")
                        print(f"Corr: {lines_file1[i]}")
                        print(f"Mine: {lines_file2[i]}")
                    print(f"Line {line_num}:")
                    print(f"Corr: {line1.strip()}")
                    print(f"Mine: {line2.strip()}")


                    continue_comparison = input("Do you want to continue comparing? (y/n): ").strip().lower()
                    if continue_comparison != 'y':
                        break

                line_num += 1

    except FileNotFoundError as e:
        print(f"Error: {e}")
1
file1_path = input("Good File: ")
file2_path = input("Your File: ")
compare_files(file1_path, file2_path)
