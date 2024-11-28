from marshmallow import Schema, fields

class PlayerSchema(Schema):
    Email = fields.Email(required=True)
    F_name = fields.Str(required=True)
    L_name = fields.Str(required=True)
    Password = fields.Str(required=True, load_only=True)
