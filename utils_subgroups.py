

OTHER = 'Other-candidates'

def get_list_group_by_dataset(dataset_name, data, sensitive_attributes, all_sensitive_attr_values):
    if dataset_name == 'law_school':

        list_protected_attributes = [['sex'], ['race'], ['race'], ['race', 'sex'], ['race', 'sex'], sensitive_attributes]
        list_attribute_group_values = [ [ 
            (['sex'],['male']), 
            (['sex'], ['female'])
        ],
        [ 
            (['race'],OTHER), 
            (['race'], ['Black'])
        ],
        [ 
            (['race'],['White']), 
            (['race'], ['Asian']),  (['race'], ['Mexican']),  (['race'], ['Hispanic']),  (['race'], ['Other']),  (['race'], ['Black']),  (['race'], ['Puertorican']),  (['race'], ['Amerindian'])
        ],
            [ 
            (['race', 'sex'], OTHER), 
            (['race', 'sex'], ['Asian', 'female']),  (['race', 'sex'], ['Mexican', 'female']),  (['race', 'sex'], ['Hispanic', 'female']),  (['race', 'sex'], ['Other', 'female']),  (['race', 'sex'], ['Black', 'female']),  (['race', 'sex'], ['Puertorican', 'female']),  (['race', 'sex'], ['Amerindian', 'female'])
        ],
            [(['race', 'sex'], ['Amerindian', 'male']), (['race', 'sex'], ['Amerindian', 'female']), (['race', 'sex'], ['Asian', 'male']), (['race', 'sex'], ['Asian', 'female']), (['race', 'sex'], ['Black', 'male']), (['race', 'sex'], ['Black', 'female']), (['race', 'sex'], ['Hispanic', 'male']), (['race', 'sex'], ['Hispanic', 'female']), (['race', 'sex'], ['Mexican', 'male']), (['race', 'sex'], ['Mexican', 'female']), (['race', 'sex'], ['Other', 'male']), (['race', 'sex'], ['Other', 'female']), (['race', 'sex'], ['Puertorican', 'male']), (['race', 'sex'], ['Puertorican', 'female']), (['race', 'sex'], ['White', 'male']), (['race', 'sex'], ['White', 'female'])]
        , all_sensitive_attr_values]
        print(len(list_protected_attributes), len(list_attribute_group_values))
        
    elif dataset_name == 'compas-poc':

        list_protected_attributes = [['sex'], ['race'], ['age_cat'], ['age_cat'], ['age_cat'], ['age_cat', 'race', 'sex'], ['age_cat', 'race', 'sex'], ['age_cat', 'race', 'sex'], ['age_cat', 'race', 'sex'], ['age_cat', 'race', 'sex'], sensitive_attributes]
        list_attribute_group_values = [ [ 
            (['sex'],['Male']), 
            (['sex'], ['Female'])
        ],
        [ 
            (['race'],['Caucasian']), 
            (['race'], ['protected'])
        ],
        [ 
            (['age_cat'],OTHER), 
            (['age_cat'], ['Less than 25'])
        ],
        [ 
            (['age_cat'],OTHER), 
            (['age_cat'], ['25 - 45'])
        ],
        [ 
            (['age_cat'],['Greater than 45']),   
            (['age_cat'], ['Less than 25']),
            (['age_cat'], ['25 - 45'])
        ],
            [ 
            (['age_cat', 'race', 'sex'], OTHER), 
            (['age_cat', 'race', 'sex'], ['Less than 25', 'protected', 'Female'])
        ],
            [ 
            (['age_cat', 'race', 'sex'], OTHER), 
            (['age_cat', 'race', 'sex'], ['Less than 25', 'Caucasian', 'Female'])
        ],
            [ 
            (['age_cat', 'race', 'sex'], OTHER), 
            (['age_cat', 'race', 'sex'], ['Less than 25', 'protected', 'Male'])
        ],
            [ 
            (['age_cat', 'race', 'sex'], OTHER), 
            (['age_cat', 'race', 'sex'], ['Less than 25', 'protected', 'Female']),
            (['age_cat', 'race', 'sex'], ['Less than 25', 'Caucasian', 'Female']),
            (['age_cat', 'race', 'sex'], ['Less than 25', 'protected', 'Male'])
        ],
    [ (['age_cat', 'race', 'sex'], OTHER), 
    (['age_cat', 'race', 'sex'],['Less than 25', 'protected', 'Male']),
    (['age_cat', 'race', 'sex'],['Less than 25', 'Caucasian', 'Male']),
    (['age_cat', 'race', 'sex'],['Less than 25', 'protected', 'Female']),
    (['age_cat', 'race', 'sex'],['Less than 25', 'Caucasian', 'Female']),
    (['age_cat', 'race', 'sex'],['25 - 45', 'protected', 'Male'])
        ],
        all_sensitive_attr_values
        ]
        print(len(list_protected_attributes), len(list_attribute_group_values))

    elif dataset_name == 'german':
        list_protected_attributes = [['sex'], ['sex', 'age']]
        list_attribute_group_values = [ [ 
            (['sex'],['male']), 
            (['sex'], ['female'])
        ],
        [ 
            (['sex', 'age'],OTHER), 
            (['sex', 'age'], ['female', 'young']), 
            (['sex', 'age'],  ['male', 'elder']),
            (['sex', 'age'],['male', 'adult'], ),
            (['sex', 'age'], ['female', 'elder']),
            (['sex', 'age'], ['female', 'adult'])
        ]]
        print(len(list_protected_attributes), len(list_attribute_group_values))
    elif dataset_name == 'artificial_1':

        sensitive_attributes_sel = sensitive_attributes[0:3]


        import itertools

        sa_values_sel = {sa: list(data[sa].unique()) for sa in sensitive_attributes_sel}
        a_sel = sa_values_sel.values()
        groups_sel = list(itertools.product(*a_sel))
        all_sensitive_attr_values_sel = [(sensitive_attributes_sel, list(group_sel)) for group_sel in groups_sel]
        list_protected_attributes = [[sa] for sa in sensitive_attributes]+ [sensitive_attributes, sensitive_attributes_sel]
        list_attribute_group_values = [[([sa],[1]), ([sa],[0])] for sa in sensitive_attributes] + [ all_sensitive_attr_values, all_sensitive_attr_values_sel]

        # Apply directly for all attributes
        list_protected_attributes = [sensitive_attributes]
        list_attribute_group_values = [all_sensitive_attr_values]

    else:
        raise ValueError

    assert len(list_protected_attributes)== len(list_attribute_group_values), 'Differ'
    return list_protected_attributes, list_attribute_group_values