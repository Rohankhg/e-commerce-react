import openpyxl
import re

FIRSTLINE = 0 #first feature line in the excel template minus 1
DEBUG = 0

class excel_template:
    'Excel Template with list of features and their lookup rules'

    def __init__(self, excel_file):
        '''
        ExcelTemplate constructor
        Reads the Excel Template to memory
        param: excelfile -> path to the excel file
        '''
        print("i just came here")
        self.workbook = openpyxl.load_workbook('TIWS CFL.xlsx')
        self.worksheet = self.workbook.active
        self.current_row = FIRSTLINE
        self.category = "System & Chassis"
        self.feature_group = ""
        self.feature = ""
        self.sub_feature = ""
        self.EOF = 0
        print(self.workbook,self.worksheet)

    def get_next_logic(self):
        "Get logic in I column"
        self.current_row = self.current_row + 1
        if self.EOF < 9:
            self.update_feature_path()
            logic = self.worksheet["I" + str(self.current_row)].value
            if logic:
                self.EOF = 0
                if re.search("P\d", logic):
                    return logic.strip()
                else:
                    return self.get_next_logic()
            else:
                self.EOF = self.EOF + 1
                return self.get_next_logic()
        else:
#            print "End Of File"
            return False

    def get_pattens(self):
        "Get patterns in column J to R"
        pattern_dict = dict()

        for col in range(10, 19):
            pattern = str(self.worksheet.cell(column=col, row=self.current_row).value)
            if pattern:
                pattern_dict["P" + str(col-9)] = pattern
            else:
                break
        return pattern_dict

    def update_feature_path(self):
        '''
        Update the feature path variables depending on content of Cells A and B of the CFL
        '''
        A = self.worksheet["A" + str(self.current_row)]
        B = self.worksheet["B" + str(self.current_row)]
        
        if not B.value and not A.value:
            #no feature detected skip
            return
        if A.value:
            if A.font.bold:
                #It's a Category
                self.category = A.value.strip()
            elif '    ' in A.value:
                #It's a Feature
                self.feature = A.value.strip()
                self.sub_feature = ""
            else:
                #It's a Feature-Group
                self.feature_group = A.value.strip()
                self.feature = ""
                self.sub_feature = ""
        if B.value:
            self.sub_feature = B.value.strip()
        return

    def get_feature_path(self):
        '''
        Returns the Feature Path composed of Category_FeatureGroup_Feature_SubFeature
        '''
        path = '{}_{}_{}'.format(self.category, self.feature_group, self.feature)
        if self.sub_feature:
            path = path + '_{}'.format(self.sub_feature)
        print(path)
        return path

    def set_result(self, result):
        "Sets the value of the status column of the excel file"
        self.worksheet["C" + str(self.current_row)].value = result

    def set_customer_name(self, customer_name):
        "Sets the customer name"
        self.worksheet["A4"].value = "at " + customer_name
        
    def set_script_version(self, script_version):
        "Sets the version of the script"
        self.worksheet["A6"].value = script_version
        
    def save_workbook(self, workbook_name):
        "Save the workbook"
        self.workbook.save(workbook_name)    



# excel=excel_template('TIWS CFL.xlsx')

template=excel_template('TIWS CFL.xlsx')
print(template)

print(excel_template.get_feature_path(template))
print(excel_template.get_next_logic(template))
print(excel_template.get_pattens(template))
print(excel_template.update_feature_path(template))