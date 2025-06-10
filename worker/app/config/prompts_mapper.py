from app.config.prompts_claude import sys_q_type_2_claude, sys_q_type_3_claude, \
    sys_q_type_3_5_claude, sys_q_type_5_claude, sys_q_type_1_5_claude, sys_q_type_6_claude, sys_q_type_7_claude, \
    sys_q_type_9_claude, sys_q_type_0_claude, sys_q_type_100_claude, sys_q_type_11_claude

from app.config.relationship_prompts import sys_q_type_1_relationship, sys_q_type_2_relationship, \
    sys_q_type_3_relationship, sys_q_type_3_5_relationship, sys_q_type_5_relationship, sys_q_type_1_5_relationship, \
    sys_q_type_6_relationship, sys_q_type_7_relationship, sys_q_type_9_relationship, sys_q_type_0_relationship, \
    sys_q_type_22_relationship, sys_q_type_11_relationship, sys_q_type_1_finetuned


class SystemPromptsMapper:
    def __init__(self):
        self.__sys_prompts_mapping_core = {
            1: sys_q_type_1_finetuned,
            2: sys_q_type_2_claude,
            152: sys_q_type_2_claude,
            3: sys_q_type_3_claude,
            35: sys_q_type_3_5_claude,
            5: sys_q_type_5_claude,
            15: sys_q_type_1_5_claude,
            6: sys_q_type_6_claude,
            7: sys_q_type_7_claude,
            9: sys_q_type_9_claude,
            0: sys_q_type_0_claude,
            100: sys_q_type_100_claude,
            11: sys_q_type_11_claude
        }

        self.__sys_prompts_mapping_relationship = {
            1: sys_q_type_1_finetuned,
            152: sys_q_type_22_relationship,
            2: sys_q_type_22_relationship,
            3: sys_q_type_3_relationship,
            35: sys_q_type_3_5_relationship,
            5: sys_q_type_5_relationship,
            15: sys_q_type_1_5_relationship,
            6: sys_q_type_6_relationship,
            7: sys_q_type_7_relationship,
            9: sys_q_type_9_relationship,
            0: sys_q_type_0_relationship,
            11: sys_q_type_11_relationship
        }

    def get_prompt(self, question_flag: int, model: str = 'core') -> str:
        if model == 'relationship':
            return self.__sys_prompts_mapping_relationship.get(question_flag, "Default system prompt")
        else:
            return self.__sys_prompts_mapping_core.get(question_flag, "Default system prompt")
