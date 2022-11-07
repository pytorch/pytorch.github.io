import json
import os


versions = os.environ['VERSIONS']
linux_matricies = os.environ['LINUX_MATRICIES']
macos_matricies = os.environ['MACOS_MATRICIES']
windows_matricies = os.environ['WINDOWS_MATRICIES']

#linux_matricies = os.getenv("LINUX_MATRICIES")
#macos_matricies = os.getenv("MACOS_MATRICIES")
#windows_matricies = os.getenv("WINDOWS_MATRICIES")

def generate_json(commands):
    matrix = '''
        {"''' + str(commands["version"]) + '''": [
          "linux": [
            "pip": [
              "accnone": [
                "note": null,
                "command": ''' + str(commands["linux_manywheel_cpu"]) + '''
              ],
              "cuda11.x": [
                "note": null,
                "command": ''' + str(commands["linux_manywheel_cuda_x"]) + '''
              ],
              "cuda11.y": [
                "note": null,
                "command": ''' + str(commands["linux_manywheel_cuda_y"]) + '''
              ],
              "rocm5.x": [
                "note": null,
                "command": ''' + str(commands["linux_manywheel_rocm"]) + '''
              ]
            ],
            "conda": [
              "cuda11.x": [
                "note": null,
                "command": ''' + str(commands["linux_conda_cuda_x"]) + '''
              ],
              "cuda11.y": [
                "note": "<b>NOTE:</b> 'conda-forge' channel is required for cudatoolkit 11.6",
                "command": ''' + str(commands["linux_conda_cuda_y"]) + '''
              ],
              "rocm5.x": [
                "note": "<b>NOTE:</b> Conda packages are not currently available for ROCm, please use pip instead<br />",
                "command": null
              ],
              "accnone": [
                "note": null,
                "command": ''' + str(commands["linux_conda_cpu"]) + '''
              ]
            ],
            "libtorch": [
              "accnone": [
                "note": null,
                "versions": [
                  "Download here (Pre-cxx11 ABI):": ''' + str(commands["linux_libtorch_cpu"][0]) + ''',
                  "Download here (cxx11 ABI):": ''' + str(commands["linux_libtorch_cpu"][1]) + '''
                ]
              ],
              "cuda11.x": [
                "note": null,
                "versions": [
                  "Download here (Pre-cxx11 ABI):": ''' + str(commands["linux_libtorch_cuda_x"][0]) + ''',
                  "Download here (cxx11 ABI):": ''' + str(commands["linux_libtorch_cuda_x"][1]) + '''
                ]
              ],
              "cuda11.y": [
                "note": null,
                "versions": [
                  "Download here (Pre-cxx11 ABI):": ''' + str(commands["linux_libtorch_cuda_y"][0]) + ''',
                  "Download here (cxx11 ABI):": ''' + str(commands["linux_libtorch_cuda_y"][1]) + '''
                ]
              ],
              "rocm5.x": [
                "note": null,
                "versions": [
                  "Download here (Pre-cxx11 ABI):": ''' + str(commands["linux_libtorch_rocm"][0]) + ''',
                  "Download here (cxx11 ABI):": ''' + str(commands["linux_libtorch_rocm"][1]) + '''
                ]
              ]
            ]
          ],
          "macos": [
            "pip": [
              "cuda11.x": [
                "note": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed",
                "command": ''' + str(commands["macos_manywheel_cuda_x"]) + '''
              ],
              "cuda11.y": [
                "note": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed",
                "command": ''' + str(commands["macos_manywheel_cuda_y"]) + '''
              ],
              "rocm5.x": [
                "note": "<b>NOTE:</b> ROCm is not available on MacOS",
                "command": null
              ],
              "accnone": [
                "note": null,
                "command": ''' + str(commands["macos_manywheel_cpu"]) + '''
              ]
            ],
            "conda": [
              "cuda11.x": [
                "note": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed",
                "command": ''' + str(commands["macos_conda_cuda_x"]) + '''
              ],
              "cuda11.y": [
                "note": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed",
                "command": ''' + str(commands["macos_conda_cuda_y"]) + '''
              ],
              "rocm5.x": [
                "note": "<b>NOTE:</b> ROCm is not available on MacOS",
                "command": null
              ],
              "accnone": [
                "note": null,
                "command": ''' + str(commands["macos_conda_cpu"]) + '''
              ]
            ],
            "libtorch": [
              "accnone": [
                "note": null,
                "versions": [
                  "Download here:": ''' + str(commands["macos_libtorch_cpu"]) + '''
                ]
              ],
              "cuda11.x": [
                "note": null,
                "versions": [
                  "MacOS binaries do not support CUDA. Download CPU libtorch here:": ''' + str(commands["macos_libtorch_cuda_x"]) + '''
                ]
              ],
              "cuda11.y": [
                "note": null,
                "versions": [
                  "MacOS binaries do not support CUDA. Download CPU libtorch here:": ''' + str(commands["macos_libtorch_cuda_y"]) + '''
                ]
              ],
              "rocm5.x": [
                "note": "<b>NOTE:</b> ROCm is not available on MacOS",
                "versions": null
              ]
            ]
          ],
          "windows": [
            "pip": [
              "accnone": [
                "note": null,
                "command": ''' + str(commands["windows_manywheel_cpu"]) + '''
              ],
              "cuda11.x": [
                "note": null,
                "command": ''' + str(commands["windows_manywheel_cuda_x"]) + '''
              ],
              "cuda11.y": [
                "note": null,
                "command": ''' + str(commands["windows_manywheel_cuda_y"]) + '''
              ],
              "rocm5.x": [
                "note": "<b>NOTE:</b> ROCm is not available on Windows",
                "command": null
              ]
            ],
            "conda": [
              "cuda11.x": [
                "note": null,
                "command": ''' + str(commands["windows_conda_cuda_x"]) + '''
              ],
              "cuda11.y": [
                "note": "<b>NOTE:</b> 'conda-forge' channel is required for cudatoolkit 11.6",
                "command": ''' + str(commands["windows_conda_cuda_y"]) + '''
              ],
              "rocm5.x": [
                "note": "<b>NOTE:</b> ROCm is not available on Windows",
                "command": null
              ],
              "accnone": [
                "note": null,
                "command": ''' + str(commands["windows_conda_cpu"]) + '''
              ]
            ],
            "libtorch": [
              "accnone": [
                "note": null,
                "versions": [
                  "Download here (Release version):": ''' + str(commands["windows_libtorch_cpu"][0]) + ''',
                  "Download here (Debug version):": ''' + str(commands["windows_libtorch_cpu"][1]) + '''
                ]
              ],
              "cuda11.x": [
                "note": null,
                "versions": [
                  "Download here (Release version):": ''' + str(commands["windows_libtorch_cuda_x"][0]) + ''',
                  "Download here (Debug version):": ''' + str(commands["windows_libtorch_cuda_x"][1]) + '''
                ]
              ],
              "cuda11.y": [
                "note": null,
                "versions": [
                  "Download here (Release version):": ''' + str(commands["windows_libtorch_cuda_y"][0]) + ''',
                  "Download here (Debug version):": ''' + str(commands["windows_libtorch_cuda_y"][1]) + '''
                ]
              ],
              "rocm5.x": [
                "note": "<b>NOTE:</b> ROCm is not available on Windows",
                "versions": null
              ]
            ]
          ]
        ]]
        '''

    return matrix


def linux_installation(matricies):

    install_dict = {}

    for matrix in matricies:
        file = open(matrix,)
        matrix_data = json.load(file)

        if matrix_data["include"][0]["package_type"] == "libtorch":
            install_dict["linux_libtorch_cpu"] = [matrix_data["include"][0]["installation"],\
                                                matrix_data["include"][1]["installation"]]
            install_dict["linux_libtorch_cuda_x"] = [matrix_data["include"][4]["installation"],\
                                                matrix_data["include"][5]["installation"]]
            install_dict["linux_libtorch_cuda_y"] = [matrix_data["include"][8]["installation"],\
                                                matrix_data["include"][9]["installation"]]
            install_dict["linux_libtorch_rocm"] = [matrix_data["include"][10]["installation"],\
                                                matrix_data["include"][11]["installation"]]

        elif matrix_data["include"][0]["package_type"] == "conda":
            install_dict["linux_conda_cpu"] = matrix_data["include"][0]["installation"]
            install_dict["linux_conda_cuda_x"] = matrix_data["include"][1]["installation"]
            install_dict["linux_conda_cuda_y"] = matrix_data["include"][2]["installation"]

        elif matrix_data["include"][0]["package_type"] == "manywheel":
            install_dict["linux_manywheel_cpu"] = matrix_data["include"][0]["installation"]
            install_dict["linux_manywheel_cuda_x"] = matrix_data["include"][1]["installation"]
            install_dict["linux_manywheel_cuda_y"] = matrix_data["include"][2]["installation"]
            install_dict["linux_manywheel_rocm"] = matrix_data["include"][3]["installation"]

        file.close()

    return install_dict


def macos_installation(matricies):

    install_dict = {}

    for matrix in matricies:
        file = open(matrix,)
        matrix_data = json.load(file)

        package = matrix_data["include"][0]["package_type"]

        install_dict["macos_" + str(package) + "_cpu"] = matrix_data["include"][0]["installation"]
        install_dict["macos_" + str(package) + "_cuda_x"] = matrix_data["include"][1]["installation"]
        install_dict["macos_" + str(package) + "_cuda_y"] = matrix_data["include"][2]["installation"]

        file.close()

    return install_dict

def windows_installation(matricies):

    install_dict = {}

    for matrix in matricies:
        file = open(matrix,)
        matrix_data = json.load(file)

        if matrix_data["include"][0]["package_type"] == "libtorch":
            install_dict["windows_libtorch_cpu"] = [matrix_data["include"][0]["installation"],\
                                                matrix_data["include"][1]["installation"]]
            install_dict["windows_libtorch_cuda_x"] = [matrix_data["include"][4]["installation"],\
                                                matrix_data["include"][5]["installation"]]
            install_dict["windows_libtorch_cuda_y"] = [matrix_data["include"][8]["installation"],\
                                                matrix_data["include"][9]["installation"]]

        else:
            package = matrix_data["include"][0]["package_type"]
            install_dict["windows_" + str(package) + "_cpu"] = matrix_data["include"][0]["installation"]
            install_dict["windows_" + str(package) + "_cuda_x"] = matrix_data["include"][1]["installation"]
            install_dict["windows_" + str(package) + "_cuda_y"] = matrix_data["include"][2]["installation"]

        file.close()

    return install_dict


def main():

    final_json = ''

    # dictionary with install insructions for all binaries
    for version in versions:
        install_dict = linux_installation(linux_matricies[version])
        install_dict.update(macos_installation(macos_matricies[version]))
        install_dict.update(windows_installation(windows_matricies[version]))
        install_dict.update('version': version)

        if len(final_json) > 0:
            final_json = final_json + ',\n' + generate_json(install_dict)

        else:
            final_json = generate_json(install_dict)

    return final_json


if __name__ == "__main__":
    file = open('../published_versions.json', 'w')
    file.write(main())
    file.close()
