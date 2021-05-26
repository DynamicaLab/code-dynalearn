class TemplateConfigTest:
    def test_attributes(self):
        for k in self.attributes:
            v = self.config.get(k)
            if v is None:
                print(f"Config with key `{k}` is `None`.")
            self.assertTrue(v is not None)

    def test_name(self):
        if "name" in self.__dict__:
            self.assertEqual(self.config.name, self.name)
