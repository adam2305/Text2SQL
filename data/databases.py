import json

class DBInfos:
    def __init__(self, tables_path):
        
        with open(tables_path, 'r') as file:
            tables = json.load(file) 
            
        self.infos = {}

        for table in tables:
            info_dict = self._load_infos(table)
            self.infos.update({table["db_id"] : info_dict})

    def __call__(self):

        return self.infos

    def _get_schemas(self, table):

        schemas_list = []
        for i,name in enumerate(table['table_names_original']):
            col_names_type = [[name[1], type] for name, type in zip(table['column_names_original'], table['column_types']) if name[0] == i]
            schemas_list.append({"name" : name, "values" : col_names_type})
        return schemas_list
        
    def _get_primary_keys(self, table):

        c_names = []
        for idx in table['primary_keys']:
            c_names.append(table['column_names_original'][idx])
        
        return [[table['table_names_original'][name[0]], name[1]] for name in c_names]
        
    def _get_foreign_keys(self, table):

        eq_list = []
        for couple in table['foreign_keys']:
            c_names = []
            for idx in couple:
                c_names.append(table['column_names_original'][idx])
            p_keys = [[table['table_names_original'][name[0]], name[1]] for name in c_names]
            eq_list.append(p_keys)
        
        return eq_list

    def _load_infos(self, table):      

        schemas = self._get_schemas(table)
        prim_keys = self._get_primary_keys(table)
        foreign_keys = self._get_foreign_keys(table)
        
        return {"schemas" : schemas, "primary_keys" : prim_keys, "foreign_keys" : foreign_keys}

    def get(self, db_id):
        """
        get infos on a database (schema, primary keys, foreign keys) from its name
        ::param db_id:: string
        """

        #check if db_id is known
        keys = self.infos.keys()
        if db_id not in keys:
            print("Unknown Key")
            return 

        return self.infos[db_id]          

def format_schema(infos):

    #schemas
    schemas = infos['schemas']
    f_schemas = []
    for schema in schemas:
        values = schema['values']
        f_values = [f"{val[0]} ({val[1]})"for val in values]
        f_schemas.append(f"{schema['name']} : {" , ".join(f_values)}")
    
    f_schemas = "\n".join(f_schemas) 

    #primary_keys
    pkeys = infos['primary_keys']
    f_pkeys = [f"{key[0]} : {key[1]}" for key in pkeys]

    f_pkeys = "\n".join(f_pkeys)

    #foreign keys
    fkeys = infos['foreign_keys']
    f_fkeys = []
    for couple in fkeys:
        keys = [f"{key[0]} : {key[1]}" for key in couple]
        f_fkeys.append(" equals ".join(keys))

    f_fkeys = "\n".join(f_fkeys)
    
    return f_schemas, f_pkeys, f_fkeys 